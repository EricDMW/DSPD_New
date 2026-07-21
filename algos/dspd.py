"""DSPD -- Distributed Primal-Dual algorithm (draft Algorithm 1, p.6).

Coupled softmax policy (Eq. 49), per-agent projected dual with schedule eta_{mu,m}=1/(2m)
(Eq. 22), and the push-sum estimator (Eqs. 19, 23, 28) over a time-varying learning network
``{G^L_m}`` that produces the Theorem-2 estimation errors E_theta(m), E_mu(m).
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Trainer
from ..policy import TabularPolicy, coupled_coupling
from ..env.topology import LearningNetwork
from ..metrics import estimation_errors


class PushSumEstimator:
    """Push-sum estimator of the manuscript, Eqs. (19), (23), (28) (25-3143_01_MS.pdf, pp. 5-6).

    Agent ``i`` maintains, for EVERY agent ``j``, an intermediate variable (``breve``) and the
    estimate (``hat``) derived from it::

        p_{i,m+1}    = sum_{l in N^L_{i,m}} w^L_{il,m} p_{l,m}                            (19a)
        mu_hat^i_j   = (1/p_{i,m+1}) sum_l w^L_{il,m} mu_breve^l_{j,m}                    (19b)
        th_hat^i_j   = (1/p_{i,m+1}) sum_l w^L_{il,m} th_breve^l_{j,m}                    (19c)

    with the weights of Eq. (1), ``w_{ij,m} = 1/|N^out_{j,m}|``, which are COLUMN-stochastic.
    After the primal/dual steps each agent revises its intermediate variable by injecting the
    increment of its own true variable, Eqs. (23)/(28) -- equivalently, in the matrix form of
    Eq. (55), ``x_breve_{j,m+1} = W_m ( x_breve_{j,m} + N (x_{j,m+1} - x_{j,m}) e_j )``::

        x_breve^i_{j,m+1} = sum_l w^L_{il,m} x_breve^l_{j,m} + w^L_{ij,m} N (x_{j,m+1} - x_{j,m})

    The ``w^L_{ij,m}`` factor is zero unless ``j`` is an in-neighbour of ``i``, which reproduces the
    two-case form of (23)/(28) exactly. This construction is what makes the scheme a *push-sum*
    rather than plain consensus: column-stochasticity preserves the invariant

        (1/N) sum_i x_breve^i_{j,m} = x_{j,m}       (stated below (23) and below (28))

    so the ratio (19b)/(19c) converges to the true value at the Theorem-2 rate O(1/m) -- the rate
    is O(1/m), not geometric, because the target itself moves with step size eta ~ O(1/m)
    (Assumption 7) while the mixing contracts geometrically (Lemma 6).

    Two deliberate, documented deviations from the manuscript, both required because the
    manuscript initialises ``theta_{j,1} = 0`` whereas the tabular softmax policy here starts from
    a randomly perturbed, idle-biased ``theta_1 != 0``:

    * the intermediate variables start at ``0`` (paper's Line 2) and the FULL initial value is
      injected as the first innovation ``x_{j,1} - 0``, which restores the invariant above at
      ``m = 1`` and reproduces the paper's transient-from-zero;
    * ``K_push`` mixing rounds may be applied per optimisation iteration instead of the paper's
      single round (Algorithm 1, Line 4). Extra rounds are pure ``W`` mixing, which preserves the
      column sums and hence the invariant; ``K_push = 1`` is the manuscript's algorithm exactly.

    NOTE: the previous implementation (run_0045 and earlier) used ROW-stochastic averaging with
    each source's diagonal clamped to its own true value. That converges too, but it is plain
    consensus with a clamped source, not the paper's push-sum: it has no ``p`` ratio correction,
    no innovation injection, and it does not preserve the average invariant. It is retained under
    ``estimator: legacy`` purely so earlier runs remain reproducible.
    """

    def __init__(self, N: int, d: int, lnet: LearningNetwork, K_push: int = 1,
                 mode: str = "pushsum", seed: int = 0):
        self.N, self.d = N, d
        self.lnet = lnet
        self.K_push = max(1, int(K_push))
        self.mode = mode
        # breve = intermediate variables (paper Line 2: all zero); hat = the derived estimates.
        self.Xb = np.zeros((N, N, d))      # th_breve^i_j
        self.Mub = np.zeros((N, N))        # mu_breve^i_j
        self.X = np.zeros((N, N, d))       # th_hat^i_j
        self.Mu = np.zeros((N, N))         # mu_hat^i_j
        self.p = np.ones(N)                # push-sum scaling, p_{i,1} = 1 (Eq. 19)
        self._prev_theta = np.zeros((N, d))
        self._prev_mu = np.zeros(N)

    def _legacy_update(self, tt, mu_true, m):
        for k in range(self.K_push):
            A = self.lnet.row_stochastic(m + k)
            self.X = np.einsum("ik,kjd->ijd", A, self.X)
            self.Mu = A @ self.Mu
            diag = np.arange(self.N)
            self.X[diag, diag, :] = tt
            self.Mu[diag, diag] = mu_true

    def update(self, theta_true: np.ndarray, mu_true: np.ndarray, m: int):
        """One optimisation iteration of the estimator; returns (E_theta, E_mu, disagreement)."""
        N, d = self.N, self.d
        tt = theta_true.reshape(N, d)

        if self.mode == "legacy":
            self._legacy_update(tt, mu_true, m)
        else:
            dth = tt - self._prev_theta                      # theta_{j,m} - theta_{j,m-1}
            dmu = mu_true - self._prev_mu
            for k in range(self.K_push):
                W = self.lnet.weight(m + k)                  # column-stochastic, Eq. (1)
                # Eq. (23)/(28): mix, then inject own increment (only on the first round).
                self.Mub = W @ self.Mub
                self.Xb = np.einsum("il,ljd->ijd", W, self.Xb)
                if k == 0:
                    self.Mub += N * W * dmu[None, :]
                    self.Xb += N * W[:, :, None] * dth[None, :, :]
                # Eq. (19a) push-sum scaling, then (19b)/(19c) ratio estimates.
                self.p = W @ self.p
                self.Mu = self.Mub / self.p[:, None]
                self.X = self.Xb / self.p[:, None, None]
            self._prev_theta, self._prev_mu = tt.copy(), np.asarray(mu_true, float).copy()

        E_theta, E_mu = estimation_errors(self.X, tt, self.Mu, mu_true)
        # Average-preservation invariant of (23)/(28); ~1e-12 when the recursion is correct.
        inv = float(np.abs(self.Mub.mean(0) - mu_true).max()) if self.mode != "legacy" else 0.0
        # disagreement: spread of the estimators about their mean (distinct from the error).
        disagree = np.linalg.norm(self.X - self.X.mean(0, keepdims=True), axis=-1).sum() / (N ** 2)
        return E_theta, E_mu, float(disagree), inv


def E_theta_sq_from(est: "PushSumEstimator", theta_true: np.ndarray) -> float:
    """(1/N^2) sum_{i,j} ||theta_hat^i_j - theta_j||^2 -- the exact quantity in the manuscript's
    Fig. 3(b) (the companion E_theta logs the un-squared norm)."""
    tt = theta_true.reshape(est.N, est.d)
    return float((np.linalg.norm(est.X - tt[None, :, :], axis=-1) ** 2).sum() / (est.N ** 2))


class DSPD(Trainer):
    algo = "dspd"

    def __init__(self, env, cfg, rng, credit_kappa: int = 1):
        kappa_p = int(cfg.get("kappa_p", 1))
        policy = TabularPolicy(env.topo, env.n_states, 5,
                               coupled_coupling(env.topo, kappa_p), seed=cfg["seed"],
                               idle_bias=float(cfg.get("init_idle_bias", 0.0)),
                               dir_bias=float(cfg.get("init_dir_bias", 0.0)))
        super().__init__(env, policy, cfg, rng, credit_kappa=credit_kappa)
        self.kappa_p = kappa_p
        self._eta_mu = float(cfg.get("eta_mu", 10.0))
        # The learning network is fixed by env_seed by default. Set lnet_per_seed to draw a
        # per-RUN network (varies across training seeds) and lnet_structure='random' for a
        # structurally-varying strongly-connected digraph -- this makes the push-sum estimation
        # error genuinely differ across seeds (visible +/-std band in E_theta/E_mu). Assumption 2
        # (uniform strong connectivity) still holds. Only affects the estimation figures.
        lnet_seed = int(cfg["seed"]) if cfg.get("lnet_per_seed", False) else int(cfg.get("env_seed", 0))
        self.lnet = LearningNetwork(env.N, seed=lnet_seed,
                                    structure=str(cfg.get("lnet_structure", "ring")),
                                    extra_edge_p=float(cfg.get("lnet_extra_edge_p", 0.06)))
        assert self.lnet.check_union_connectivity(self.lnet.union_window), \
            "learning net not uniformly strongly connected over its switching period (Assumption 6)"
        d = env.n_states * 5
        # 'pushsum' (default) = manuscript Eqs. (19)/(23)/(28); 'legacy' = the row-stochastic
        # clamped-source consensus used up to run_0046 (kept only for reproducing those runs).
        self.estimator = PushSumEstimator(env.N, d, self.lnet, K_push=int(cfg.get("K_push", 1)),
                                          mode=str(cfg.get("estimator", "pushsum")))
        self._last_est = {}

    def eta_mu(self, m: int) -> float:
        # Two dual step-size modes (brief section 10 allows "schedule vs fixed"):
        #  * 'fixed'   (default): equilibrating dual with fixed eta_mu -> finds the binding
        #    point and matches the asymptotic objective (the paper's claim a). The DSPD
        #    speed-up over SPDAC then comes from the COUPLED policy, not the dual schedule.
        #  * 'inverse': the draft's theoretical eta_{mu,m}=1/(2m) (used in the section-9.5
        #    dual step-size sweep). In finite horizons this over-accumulates mu, so it is not
        #    the default (documented in logs/verification_report.md).
        #  * 'paper'   (lr_schedule=paper): the draft's EXACT theoretical dual step
        #    eta_{mu,m}=1/(2m) (Algorithm 1 / Theorem 3), independent of eta_mu.
        if self.cfg.get("lr_schedule", "current") == "paper":
            return 1.0 / (2.0 * m)
        if self.cfg.get("dual_schedule", "fixed") == "inverse":
            return self._eta_mu / (2.0 * m)
        return self._eta_mu

    def dual_update(self, G_i: np.ndarray, m: int):
        # projected descent of L in mu (Eq. 22): mu <- P_[0,mu_max]( mu - eta*(G_i - c_i) )
        eta = self.eta_mu(m)
        self.mu = np.clip(self.mu - eta * (G_i - self.c_i), 0.0, self.mu_max)

    def iterate(self, m: int) -> dict:
        info = super().iterate(m)
        theta = self.policy.theta.detach().cpu().numpy()
        E_theta, E_mu, disagree, inv = self.estimator.update(theta, self.mu, m)
        self._last_est = {"E_theta": E_theta, "E_mu": E_mu, "pushsum_disagreement": disagree,
                          # Fig. 3 of the manuscript plots the SQUARED parameter error
                          # (1/N^2) sum_{i,j} ||th_hat^i_j - th_j||^2; log both forms.
                          "E_theta_sq": E_theta_sq_from(self.estimator, theta),
                          "pushsum_invariant_err": inv}
        info.update(self._last_est)
        return info

    def evaluate(self, M: int) -> dict:
        out = super().evaluate(M)
        out.update(self._last_est)  # attach latest estimation diagnostics to the checkpoint
        return out
