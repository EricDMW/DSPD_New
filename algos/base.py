"""Shared training machinery for the tabular primal-dual methods (brief section 8).

All methods share the same environment, constraint (brief section 5), gamma, H and threshold.
The policy gradient is a Monte-Carlo REINFORCE estimator of the truncated coupled policy
gradient (draft Eq. 16/18): each agent j's score log pi_j is weighted by the
kappa-hop-neighbourhood-averaged Lagrangian return-to-go (the truncated-Q surrogate, matching
the reference ``average_score = neighbor_score_sum / n_neighbor_dict``). Because the policy's
theta is a single autograd tensor, differentiating the surrogate yields the coupled gradient
for DSPD and the factorized gradient for SPDAC/MAPPO-L automatically (brief section 14.17).

Variance reduction: a time-dependent batch-mean baseline b_{i,t} = mean_b A_{i,t,b} is
subtracted (a function of t only, not of the sampled action -> unbiased).
"""

from __future__ import annotations

import numpy as np
import torch

from ..env.topology import Topology


def neighbor_weight_matrix(topo: Topology, kappa: int, mode: str = "sum") -> np.ndarray:
    """Credit-assignment weights W[i,j] for j in N_i^{E,kappa} (incl. self), else 0.  (N, N).

    Two normalisations of the kappa-hop return aggregation used as agent i's policy-gradient
    credit A_i = sum_j W[i,j] R_j:

    * ``mode="sum"`` (draft Eq. (38), default): W[i,j] = 1/N for j in N_i^{E,kappa}. This is the
      truncated policy gradient of the manuscript, which approximates the *global* Lagrangian
      Q-function Q^pi by (1/N) sum_i Q_{tru,i} -- a kappa-INDEPENDENT (global) normalisation.
      A larger truncation radius kappa therefore ACCUMULATES more of the (correlation-decaying)
      returns that theta_i genuinely influences, reducing the O(gamma^{h(kappa,kappa_p)+1})
      truncation bias of Lemma 2 (Eq. (37)); larger kappa helps, as the theory predicts.

    * ``mode="average"`` (legacy, matches the non-authoritative reference
      ``average_score = neighbor_score_sum / n_neighbor_dict``): W[i,j] = 1/|N_i^{E,kappa}|.
      The per-neighbourhood normalisation shrinks with kappa, so a larger radius DILUTES the
      differentiated local credit toward the global mean return and empirically *hurts* -- an
      artefact of the normalisation, not of the truncation.
    """
    N = topo.N
    nb = topo.khop(kappa)
    W = np.zeros((N, N))
    for i in range(N):
        for j in nb[i]:
            W[i, j] = 1.0
        if mode == "average":
            W[i] /= W[i].sum()
        elif mode == "sum":
            W[i] /= N           # global (1/N) normalisation, draft Eq. (38)
        else:
            raise ValueError(f"unknown credit mode {mode!r}")
    return W


def returns_to_go(f_HBN: np.ndarray, g_HBN: np.ndarray, mu: np.ndarray, gamma: float) -> np.ndarray:
    """Lagrangian-combined discounted return-to-go R_{i,t} = Sum_{t'>=t} gamma^{t'-t} (f + mu*g).

    mu: (N,) current dual (detached). Returns (H, B, N).
    """
    H, B, N = f_HBN.shape
    r = f_HBN + mu[None, None, :] * g_HBN
    R = np.empty_like(r)
    running = np.zeros((B, N))
    for t in reversed(range(H)):
        running = r[t] + gamma * running
        R[t] = running
    return R


class Trainer:
    """Base primal-dual trainer. Subclasses set the policy coupling and the dual rule."""

    algo = "base"

    def __init__(self, env, policy, cfg, rng: np.random.Generator, credit_kappa: int = 1):
        self.env = env
        self.topo = env.topo
        self.policy = policy
        self.cfg = cfg
        self.rng = rng
        self.gamma = cfg["gamma"]
        self.H = env.H
        self.N = env.N
        self.credit_kappa = credit_kappa
        self.mu = np.zeros(self.N, dtype=np.float64)          # duals mu_i >= 0
        self.mu_max = float(cfg.get("mu_max", 100.0))
        self.c_i = np.full(self.N, cfg["c_i"], dtype=np.float64)
        # credit_norm = truncated-gradient normalisation ('sum'=draft Eq.38 | 'average'=legacy).
        # NB distinct from MAPPO-L's variant 'credit_mode' ('self'/'neigh'); do not conflate.
        self.credit_norm = cfg.get("credit_norm", "sum")
        self.Wk = torch.as_tensor(neighbor_weight_matrix(self.topo, credit_kappa, self.credit_norm))
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg["actor_lr"])
        self.base_lr = cfg["actor_lr"]
        # optional diminishing policy step-size eta_theta,m = base_lr / (1 + m/tau_lr).
        # DSPD uses this (matched to its diminishing dual, draft eta_theta,m ~ 1/(2m+L_theta));
        # baselines keep a fixed step (brief section 14.22 tuning parity is per-family).
        self.lr_tau = cfg.get("lr_tau", None)
        # Policy step-size schedule selector (brief section 10):
        #   'current' (default): the tuned diminishing form base_lr/(1+m/lr_tau) above.
        #   'paper'  : the draft's EXACT theoretical step eta_theta,m = 1/(2m + L_theta)
        #             (Algorithm 1 / Theorem 3, Assumption 7). L_theta is the paper's Lipschitz
        #             constant L_{theta theta}; its closed form is impractically large for this
        #             env, so it is exposed as cfg['lr_Ltheta'] (set to keep the initial step
        #             ~ the tuned rate). Driven as the Adam lr, matching the 'current' mechanism.
        self.lr_schedule = str(cfg.get("lr_schedule", "current"))
        self.lr_Ltheta = float(cfg.get("lr_Ltheta", 80.0))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.env_steps = 0

    def _set_lr(self, m: int):
        if self.lr_schedule == "paper":
            lr = 1.0 / (2.0 * m + self.lr_Ltheta)          # eta_theta,m = 1/(2m + L_theta)
            for grp in self.opt.param_groups:
                grp["lr"] = lr
        elif self.lr_tau is not None:
            lr = self.base_lr / (1.0 + m / float(self.lr_tau))
            for grp in self.opt.param_groups:
                grp["lr"] = lr

    # -- sampling helpers -------------------------------------------------------------
    def _policy_sample(self):
        return lambda s: self.policy.sample(s)

    def collect(self, B: int) -> dict:
        roll = self.env.rollout(self._policy_sample(), B)
        # environment transitions (each of B episodes advances H steps); the primary x-axis
        # (brief section 11.4). Same convention for every method -> curves are comparable.
        self.env_steps += roll["H"] * B
        return roll

    # -- policy gradient (REINFORCE surrogate) ----------------------------------------
    def _advantage(self, roll: dict) -> torch.Tensor:
        """kappa-hop-neighbourhood-averaged Lagrangian return-to-go, baselined. (H,B,N)."""
        R = returns_to_go(roll["f"], roll["g"], self.mu, self.gamma)   # (H,B,N)
        Rt = torch.as_tensor(R)
        A = torch.einsum("hbj,ij->hbi", Rt, self.Wk)                    # neighbourhood average
        A = A - A.mean(dim=1, keepdim=True)                            # time-dependent baseline
        return A

    def probe_grad_norm(self, B: int) -> float:
        """Low-variance measurement of the true policy-gradient norm ||grad_theta L|| at a large
        batch B, at the CURRENT (theta, mu). Pure instrumentation: takes NO optimiser step and does
        NOT perturb the training trajectory -- the env RNG, global torch RNG and env_step counter are
        snapshotted and restored, so an inline probe leaves the training sequence bit-identical.
        Reducing the Monte-Carlo variance (~1/B) isolates the truncation-limited residual of
        Theorem 3 (E = O(eps) + O(eps'(kappa,kappa_p))) from finite-sample gradient noise."""
        env_state = self.env.rng.bit_generator.state
        torch_state = torch.get_rng_state()
        saved_steps = self.env_steps
        try:
            roll = self.collect(B)
            A = self._advantage(roll).detach()
            states = torch.as_tensor(roll["states"], dtype=torch.long)
            actions = torch.as_tensor(roll["actions"], dtype=torch.long)
            H, Bn, N = states.shape
            logp = self.policy.action_logprobs(states.reshape(H * Bn, N), actions.reshape(H * Bn, N))
            loss = -(A.reshape(H * Bn, N) * logp).sum(dim=1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            sq = sum((p.grad.detach() ** 2).sum() for p in self.policy.parameters() if p.grad is not None)
            gn = float(torch.sqrt(sq))
            self.opt.zero_grad(set_to_none=True)
        finally:
            self.env.rng.bit_generator.state = env_state
            torch.set_rng_state(torch_state)
            self.env_steps = saved_steps
        return gn

    def policy_update(self, roll: dict):
        states = torch.as_tensor(roll["states"], dtype=torch.long)     # (H,B,N)
        actions = torch.as_tensor(roll["actions"], dtype=torch.long)
        A = self._advantage(roll).detach()
        H, B, N = states.shape
        s_flat = states.reshape(H * B, N)
        a_flat = actions.reshape(H * B, N)
        A_flat = A.reshape(H * B, N)
        # K_theta policy-gradient steps per iteration on the collected batch (draft Alg. 1,
        # Lines 12-16). Matches the per-iteration optimisation budget of the PPO-based MAPPO-L
        # baseline (its ``update_epochs``), so the comparison is fair; K_theta=1 recovers the
        # plain single-step estimator. surrogate: -(1/HB) sum_{t,b} sum_j A_j log pi_j(a_j|s_j).
        K = int(self.cfg.get("update_epochs", 1))
        loss_val, gnorm = 0.0, 0.0
        for _ in range(K):
            logp = self.policy.action_logprobs(s_flat, a_flat)
            loss = -(A_flat * logp).sum(dim=1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm))
            self.opt.step()
            loss_val = float(loss.item())
        assert torch.isfinite(self.policy.theta).all(), "NaN/Inf in policy params"  # section 14.26
        return loss_val, gnorm

    # -- dual update (subclass overrides schedule/form) -------------------------------
    def dual_update(self, G_i: np.ndarray, m: int):
        raise NotImplementedError

    def eta_mu(self, m: int) -> float:
        raise NotImplementedError

    # -- one optimisation iteration ---------------------------------------------------
    def iterate(self, m: int) -> dict:
        self._set_lr(m)
        roll = self.collect(self.cfg["n_sample_traj"])
        # dual uses a fresh constraint estimate from this batch's discounted G_i
        G_i = (self.gamma ** np.arange(self.H))[:, None, None]
        G_i = (G_i * roll["g"]).sum(0).mean(0)                         # (N,)
        loss, gnorm = self.policy_update(roll)
        self.dual_update(G_i, m)
        return {"policy_loss": loss, "actor_grad_norm": gnorm}

    # -- evaluation (fresh eval batch, brief section 6/14.19) -------------------------
    def evaluate(self, M: int) -> dict:
        from ..metrics import compute_metrics
        roll = self.env.rollout(self._policy_sample(), M)
        return compute_metrics(roll, self.gamma, self.c_i, mu=self.mu)

    # -- checkpoint payload -----------------------------------------------------------
    def state_dict(self):
        return {
            "theta": self.policy.theta.detach().cpu().numpy(),
            "mu": self.mu.copy(),
            "env_steps": self.env_steps,
        }
