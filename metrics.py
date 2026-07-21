"""Monte-Carlo estimators for objective, constraint and violation (brief section 6).

All expectations are estimated from a **fresh, fixed-size** evaluation batch of ``M``
trajectories (brief section 6 / section 14.19: never reuse the training batch). The *same*
eval batch is used for objective and constraint at a given checkpoint.

Sign convention (brief section 5 / section 14.3): ``g_i, G_i, c_i <= 0`` and the constraint
is ``G_i >= c_i``; violation is ``max(0, c_i - G_i)``. This is unit-tested.
"""

from __future__ import annotations

import numpy as np

from .env.topology import N_ACTIONS, IDLE


def threshold_c(beta: float, gamma: float, H: int) -> float:
    """Finite-horizon budget threshold c_i = -beta*(1-gamma^H)/(1-gamma) (brief section 5)."""
    return -beta * (1.0 - gamma ** H) / (1.0 - gamma)


def constraint_bounds(gamma: float, H: int):
    """G_i lies in [-(1-gamma^H)/(1-gamma), 0]."""
    return -(1.0 - gamma ** H) / (1.0 - gamma), 0.0


def discounted_returns(reward_HBN: np.ndarray, gamma: float) -> np.ndarray:
    """Sum_t gamma^t r(t) per (trajectory, agent). reward: (H,B,N) -> (B,N)."""
    H = reward_HBN.shape[0]
    disc = (gamma ** np.arange(H))[:, None, None]
    return (disc * reward_HBN).sum(axis=0)


def discounted_occupancy(actions_HBN: np.ndarray, gamma: float, n_actions: int = N_ACTIONS) -> np.ndarray:
    """lambda_i(a) = E[Sum_t gamma^t 1[a_{i,t}=a]], averaged over trajectories. -> (N, |A|)."""
    H, B, N = actions_HBN.shape
    disc = (gamma ** np.arange(H))[:, None, None]
    onehot = np.zeros((H, B, N, n_actions))
    np.put_along_axis(onehot, actions_HBN[..., None], 1.0, axis=-1)
    lam = (disc[..., None] * onehot).sum(axis=0).mean(axis=0)  # (N, |A|)
    return lam


def compute_metrics(rollout: dict, gamma: float, c_i: np.ndarray, mu: np.ndarray | None = None) -> dict:
    """Compute all section-6 metrics from an evaluation rollout.

    Returns a dict with scalar means and per-agent arrays (keys ending ``_per_agent``).
    """
    f, g, a = rollout["f"], rollout["g"], rollout["actions"]
    N = f.shape[-1]

    F_traj = discounted_returns(f, gamma)          # (B, N)
    G_traj = discounted_returns(g, gamma)          # (B, N)
    F_i = F_traj.mean(axis=0)                       # (N,)  F_hat_i
    G_i = G_traj.mean(axis=0)                       # (N,)  G_hat_i
    F_undisc = f.sum(axis=0).mean(axis=0)           # (N,)  undiscounted success count

    slack = G_i - c_i                               # >= 0 feasible
    feasible = (G_i >= c_i).astype(np.float64)
    short = np.clip(c_i - G_i, 0.0, None)           # per-agent positive shortfall
    violation_total = float(short.sum())            # summed clamp (brief section 6, matches ref)
    violation_mean = violation_total / N
    violation_max = float(short.max())

    # Across-AGENT dispersion of the constraint return. The mean G_i.mean() can sit far above
    # c_i (feasible on average) while the *worst* agent dips below it -- exactly the source of a
    # nonzero violation_total. Logging the worst / best agent lets the constraint figure show the
    # per-agent range rather than only the (misleadingly comfortable) mean.
    G_min = float(G_i.min())                        # worst agent = argmin_i G_i (binds c_i)
    G_max = float(G_i.max())                        # best agent
    G_std_agents = float(G_i.std())                 # spread across agents

    lam = discounted_occupancy(a, gamma)            # (N, |A|)
    transmit_freq = lam[:, 1:].sum(axis=1)          # Sum_{a != Idle} lambda_i(a)  (Idle=0)

    out = {
        "objective_F": float(F_i.mean()),
        "objective_F_undisc": float(F_undisc.mean()),
        "constraint_G": float(G_i.mean()),
        "constraint_G_min": G_min,                  # worst-agent constraint return
        "constraint_G_max": G_max,                  # best-agent constraint return
        "constraint_G_std_agents": G_std_agents,    # across-agent std
        "constraint_slack": float(slack.mean()),
        "feasible_frac": float(feasible.mean()),
        "violation_total": violation_total,
        "violation_mean": violation_mean,
        "violation_max": violation_max,
        "transmit_freq": float(transmit_freq.mean()),
        "end_step": float(rollout["H"]),
        # per-agent arrays (logged with scope='agent')
        "objective_F_per_agent": F_i,
        "constraint_G_per_agent": G_i,
        "transmit_freq_per_agent": transmit_freq,
        "occupancy_lambda_per_agent": transmit_freq,  # scalar summary per agent
    }
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float64)
        lagr = float(F_i.mean() + (mu * (G_i - c_i)).mean())
        out["dual_mu"] = float(mu.mean())
        out["lagrangian_L"] = lagr
        out["dual_mu_per_agent"] = mu
    return out


def estimation_errors(theta_hat: np.ndarray, theta_true: np.ndarray,
                      mu_hat: np.ndarray, mu_true: np.ndarray) -> tuple[float, float]:
    """DSPD Theorem-2 errors (brief section 6).

    theta_hat: (N, N, d) estimate by agent i of agent j's params; theta_true: (N, d).
    E_theta = (1/N^2) Sum_{i,j} ||theta_hat[i,j] - theta_true[j]||_2
    E_mu    = (1/N^2) Sum_{i,j} |mu_hat[i,j] - mu_true[j]|
    """
    N = theta_true.shape[0]
    dtheta = theta_hat - theta_true[None, :, :].reshape(1, N, -1)
    E_theta = np.linalg.norm(dtheta, axis=-1).sum() / (N * N)
    E_mu = np.abs(mu_hat - mu_true[None, :]).sum() / (N * N)
    return float(E_theta), float(E_mu)
