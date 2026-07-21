"""Matched tabular softmax policy: coupled (DSPD) vs. factorized (SPDAC / MAPPO-L).

brief section 8 / section 14.17: DSPD and SPDAC must share the *same* tabular policy class
(|S_i|*|A| = 4*5 = 20 logits per agent) and differ *only* in coupled vs. factorized
parameterisation. We realise both as a linear mix of the shared parameter tensor
``theta`` (N, |S|, |A|) through an (N, N) coupling matrix ``C``:

    logits_i(s, a) = sum_j C[i, j] * theta[j, s, a]

* factorized (SPDAC / MAPPO-L):  C = I_N               (logits_i = theta_i)
* coupled (DSPD, draft Eq. 49):  C[i,i] = 0.9,
                                 C[i,j] = 0.1 / |N_{i,-i}^{E,kappa_p}|  for kappa_p-neighbours,
                                 C[i,i] = 1.0 if agent i has no kappa_p-neighbour (brief section 14.11).

Because ``theta`` is a single autograd tensor, ``d/d theta_i log pi_j`` is non-zero exactly
when i couples into j (coupled) and zero otherwise (factorized) -- so the coupled-vs-
independent policy gradient (draft Eq. 16/18) is computed *exactly*, not approximated.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .env.topology import Topology


def factorized_coupling(N: int) -> torch.Tensor:
    return torch.eye(N, dtype=torch.float64)


def coupled_coupling(topo: Topology, kappa_p: int = 1) -> torch.Tensor:
    """Draft Eq. (49) coupling matrix over the kappa_p-hop environmental neighbourhood."""
    N = topo.N
    C = torch.zeros((N, N), dtype=torch.float64)
    nbrs = topo.khop_excl_self(kappa_p)
    for i in range(N):
        nb = nbrs[i]
        if len(nb) == 0:  # div-by-zero guard (brief section 14.11): drop coupling term
            C[i, i] = 1.0
            continue
        C[i, i] = 0.9
        for j in nb:
            C[i, j] = 0.1 / len(nb)
    return C


class TabularPolicy(torch.nn.Module):
    """Softmax policy with shared parameter tensor and a fixed coupling matrix."""

    def __init__(self, topo: Topology, n_states: int, n_actions: int,
                 coupling: torch.Tensor, seed: int = 0, idle_bias: float = 0.0,
                 dir_bias: float = 0.0, dir_action: int = 1):
        super().__init__()
        self.topo = topo
        self.N = topo.N
        self.n_states = n_states
        self.n_actions = n_actions
        g = torch.Generator().manual_seed(seed)
        # Small random init (draft Section VI initialises theta near a fixed point).
        #  * idle_bias > 0 biases the Idle action so the initial policy is FEASIBLE (few
        #    transmissions => G_i ~ 0 >= c_i); learning then increases F while G descends to c_i.
        #  * dir_bias > 0 biases a single directional action so every agent starts TRANSMITTING
        #    HARD toward the same corner (over the budget => initially INFEASIBLE); the primal-dual
        #    dual then drives the constraint violation down to feasibility over training (the
        #    safe-RL "first violate, then satisfy" curve). Use one or the other, not both.
        theta0 = 0.01 * torch.randn((self.N, n_states, n_actions), generator=g, dtype=torch.float64)
        from .env.topology import IDLE
        theta0[:, :, IDLE] += float(idle_bias)
        if dir_bias:
            theta0[:, :, int(dir_action)] += float(dir_bias)
        self.theta = torch.nn.Parameter(theta0)
        self.register_buffer("C", coupling.to(torch.float64))

    def logits_all(self) -> torch.Tensor:
        """(N, |S|, |A|) coupled logits = einsum('ij,jsa->isa', C, theta)."""
        return torch.einsum("ij,jsa->isa", self.C, self.theta)

    def dist_logits(self, states: torch.Tensor) -> torch.Tensor:
        """Gather per-(sample, agent) logits. states: (B, N) long -> (B, N, |A|)."""
        logits = self.logits_all()                      # (N, |S|, |A|)
        idx = states.unsqueeze(-1).expand(-1, -1, self.n_actions)  # (B, N, |A|)
        # logits[i, states[b,i], :]  ->  gather along the state dim per agent
        per_agent = logits.permute(0, 1, 2)             # (N, |S|, |A|)
        gathered = per_agent[torch.arange(self.N).unsqueeze(0), states]  # (B, N, |A|)
        return gathered

    def action_logprobs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log pi(a|s) for given (states, actions), shape (B, N), differentiable in theta."""
        lg = self.dist_logits(states)                   # (B, N, |A|)
        logp = F.log_softmax(lg, dim=-1)
        return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, states_np: np.ndarray) -> np.ndarray:
        """Sample actions for numpy state indices (B, N) -> actions (B, N) numpy."""
        states = torch.as_tensor(states_np, dtype=torch.long)
        lg = self.dist_logits(states)
        probs = F.softmax(lg, dim=-1)
        B, N, A = probs.shape
        flat = probs.reshape(-1, A)
        a = torch.multinomial(flat, 1).reshape(B, N)
        return a.numpy()

    @torch.no_grad()
    def action_probs_by_state(self) -> np.ndarray:
        """pi(.|s) for every (agent, state): (N, |S|, |A|). Used for occupancy metrics."""
        lg = self.logits_all()
        return F.softmax(lg, dim=-1).numpy()
