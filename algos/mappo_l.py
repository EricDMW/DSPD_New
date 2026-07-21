"""MAPPO-L -- multi-agent PPO with a Lagrangian on a discounted cost (Gu et al.).

External baseline (brief section 8). PPO-clip actor on the matched factorized tabular policy;
Lagrangian dual on the **discounted** (same gamma) cost 1[a_i != Idle] with budget d_i = -c_i
(brief section 14.5: forced to the same discounted-sum convention). Combined Lagrangian reward
f + lambda*g reuses the base neighbourhood-averaged advantage. Dual is clipped/normalised for
stability (brief section 14.16).

Three variants (brief section 8/9):
  * ``mappo_l``          centralized:              shared dual,      neighbourhood-credit advantage
  * ``mappo_l_dec``      decentralized:            per-agent dual,   self-credit advantage
  * ``mappo_l_decagg``   decentralized-aggregate:  per-agent dual,   neighbourhood-credit advantage
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Trainer
from ..policy import TabularPolicy, factorized_coupling

# (dual_mode, credit_mode). The safety constraint of problem (4) is PER-AGENT (G_i >= c_i for
# every i), so a single shared dual on the *average* constraint cannot enforce it -- it stays
# slack whenever the mean is feasible while individual agents overshoot, and the policy then
# violates without bound (an unfair strawman). All variants therefore use PER-AGENT duals mu_i,
# a proper MAPPO-Lagrangian for per-agent constraints; the variants differ only in the
# information each agent uses for its credit (global / neighbourhood / self), mirroring the
# centralized-vs-decentralized distinction of Gu et al.
_VARIANTS = {
    "mappo_l": ("per_agent", "global"),        # centralized: global information (all agents)
    "mappo_l_dec": ("per_agent", "self"),      # decentralized: local, self credit
    "mappo_l_decagg": ("per_agent", "neigh"),  # decentralized-aggregate: local neighbourhood
}


class MAPPOL(Trainer):
    def __init__(self, env, cfg, rng, variant: str = "mappo_l", credit_kappa: int = 1):
        assert variant in _VARIANTS, f"unknown MAPPO-L variant {variant}"
        self.variant = variant
        self.algo = variant
        self.dual_mode, self.credit_mode = _VARIANTS[variant]
        policy = TabularPolicy(env.topo, env.n_states, 5,
                               factorized_coupling(env.N), seed=cfg["seed"],
                               idle_bias=float(cfg.get("init_idle_bias", 0.0)),
                               dir_bias=float(cfg.get("init_dir_bias", 0.0)))
        # credit overrides (kappa/neigh handled by the base Wk); set after super().__init__
        super().__init__(env, policy, cfg, rng, credit_kappa=credit_kappa)
        if self.credit_mode == "self":
            self.Wk = torch.eye(env.N, dtype=torch.float64)              # each agent: own return
        elif self.credit_mode == "global":
            self.Wk = torch.full((env.N, env.N), 1.0 / env.N, dtype=torch.float64)  # team return
        self._eta_mu = float(cfg.get("eta_mu", 2.0))
        self.clip_eps = float(cfg.get("clip_eps", 0.2))
        self.update_epochs = int(cfg.get("update_epochs", 1))

    def eta_mu(self, m: int) -> float:
        if self.cfg.get("lr_schedule", "current") == "paper":
            return 1.0 / (2.0 * m)  # draft eta_{mu,m}=1/(2m) (Alg. 1 / Thm 3)
        return self._eta_mu

    def dual_update(self, G_i: np.ndarray, m: int):
        # cost C_i = -G_i, budget d_i = -c_i; ascent lambda <- clip(lambda + eta*(C_i - d_i))
        #   = clip(lambda + eta*(c_i - G_i)) : rises exactly when G_i < c_i (violated).
        eta = self.eta_mu(m)
        step = self.c_i - G_i
        if self.dual_mode == "shared":
            self.mu = np.clip(self.mu + eta * step.mean(), 0.0, self.mu_max)
        else:
            self.mu = np.clip(self.mu + eta * step, 0.0, self.mu_max)

    def policy_update(self, roll: dict):
        states = torch.as_tensor(roll["states"], dtype=torch.long)
        actions = torch.as_tensor(roll["actions"], dtype=torch.long)
        A = self._advantage(roll).detach()
        H, B, N = states.shape
        s_flat = states.reshape(H * B, N)
        a_flat = actions.reshape(H * B, N)
        A_flat = A.reshape(H * B, N)
        with torch.no_grad():
            old_logp = self.policy.action_logprobs(s_flat, a_flat)
        last_loss, last_gnorm = 0.0, 0.0
        for _ in range(self.update_epochs):
            logp = self.policy.action_logprobs(s_flat, a_flat)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * A_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * A_flat
            loss = -torch.minimum(surr1, surr2).sum(dim=1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt.step()
            last_loss, last_gnorm = float(loss.item()), float(gnorm)
        assert torch.isfinite(self.policy.theta).all(), "NaN/Inf in policy params"
        return last_loss, last_gnorm
