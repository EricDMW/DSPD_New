"""SPDAC -- Scalable Primal-Dual Actor-Critic, linear/tabular case ([spdac] Algorithm 1).

Matched tabular policy class (brief section 8) but **factorized** (no coupling) -- the only
difference from DSPD. In the linear case the shadow reward degenerates to the *constant*
u_i = g_i, so the constraint value is <u_i, lambda_i> = G_i (the discounted constraint
return); we therefore do NOT run the general-utility occupancy-gradient path (brief section
14.15). REINFORCE policy update with a fixed dual step-size (brief section 10, baselines fixed).
"""

from __future__ import annotations

import numpy as np

from .base import Trainer
from ..policy import TabularPolicy, factorized_coupling


class SPDAC(Trainer):
    algo = "spdac"

    def __init__(self, env, cfg, rng, credit_kappa: int = 1):
        policy = TabularPolicy(env.topo, env.n_states, 5,
                               factorized_coupling(env.N), seed=cfg["seed"],
                               idle_bias=float(cfg.get("init_idle_bias", 0.0)),
                               dir_bias=float(cfg.get("init_dir_bias", 0.0)))
        super().__init__(env, policy, cfg, rng, credit_kappa=credit_kappa)
        self._eta_mu = float(cfg.get("eta_mu", 2.0))

    def eta_mu(self, m: int) -> float:
        if self.cfg.get("lr_schedule", "current") == "paper":
            return 1.0 / (2.0 * m)  # draft eta_{mu,m}=1/(2m) (Alg. 1 / Thm 3)
        return self._eta_mu  # fixed dual step-size (baseline)

    def dual_update(self, G_i: np.ndarray, m: int):
        # <u_i, lambda_i> - c_i = G_i - c_i ; projected descent (same sign convention as DSPD).
        self.mu = np.clip(self.mu - self.eta_mu(m) * (G_i - self.c_i), 0.0, self.mu_max)
