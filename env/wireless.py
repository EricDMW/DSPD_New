"""Vectorised 5x5 wireless access-control environment (brief section 4; SPDAC App. H.3).

Reimplemented from the papers, verified against the reference
``Decentralized-Safe-MARL-with-General-Utilities/envs/wireless_comm.py``. The dynamics are
identical to the reference; only the *constraint* (cumulative transmission budget, brief
section 5) and the algorithms are implemented independently (brief section 2).

The environment runs a **batch of B independent episodes** vectorised over numpy for speed
(tabular policies, N=25, H=12). Each agent's local state is 2 deadline bits -> |S_i|=4.
"""

from __future__ import annotations

import numpy as np

from .topology import Topology, N_ACTIONS, IDLE


class WirelessEnv:
    """Batched 5x5 wireless env. All expectations are Monte-Carlo over the B episodes.

    Parameters are the *fixed instance* (brief section 4): ``p`` per-agent arrival probs and
    ``q`` per-AP success probs are sampled once by ``env_seed`` and reused across all algos
    and training seeds. The per-rollout stochastic draws use a separate ``rng`` seeded by the
    training seed (Monte-Carlo noise varies by seed; the instance does not).
    """

    def __init__(self, topo: Topology, p: np.ndarray, q: np.ndarray,
                 ddl: int = 2, H: int = 12, rng: np.random.Generator | None = None):
        self.topo = topo
        self.N = topo.N
        self.ddl = ddl
        self.H = H
        self.n_states = 2 ** ddl  # |S_i| = 4
        self.p = np.asarray(p, dtype=np.float64)          # (N,)
        self.q = np.asarray(q, dtype=np.float64).reshape(-1)  # (n_ap,)
        assert self.p.shape == (self.N,)
        assert self.q.shape == (topo.n_ap,)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # deadline weight for state index: bit k contributes 2**k -> index in [0, 4)
        self._bitw = (2 ** np.arange(ddl)).astype(np.int64)

    # -- instance construction --------------------------------------------------------
    @classmethod
    def fixed_instance(cls, env_seed: int, L: int = 5, ddl: int = 2, H: int = 12,
                       pkg_p: float = 0.5, success_p: float = 0.8,
                       heterogeneous: bool = False, het_spread: float = 0.1,
                       rng: np.random.Generator | None = None):
        """Sample the fixed instance once (brief section 4, section 14.10 no-leakage rule)."""
        topo = Topology(L)
        inst_rng = np.random.default_rng(env_seed)
        if heterogeneous:
            p = np.clip(pkg_p + inst_rng.uniform(-het_spread, het_spread, topo.N), 0.05, 0.95)
            q = np.clip(success_p + inst_rng.uniform(-het_spread, het_spread, topo.n_ap), 0.05, 0.95)
        else:
            # Default homogeneous instance == the reference (brief decision, spec_extracted.md).
            p = np.full(topo.N, pkg_p, dtype=np.float64)
            q = np.full(topo.n_ap, success_p, dtype=np.float64)
        return cls(topo, p, q, ddl=ddl, H=H, rng=rng)

    def instance_dict(self):
        """Everything needed to persist the fixed instance (brief section 11.1 env_instance)."""
        return dict(
            L=self.topo.L, N=self.N, ddl=self.ddl, H=self.H,
            n_ap=self.topo.n_ap, p=self.p, q=self.q,
            adjacency=self.topo.adj.astype(np.int8),
            ap_target=self.topo.ap_target,
        )

    # -- rollout ----------------------------------------------------------------------
    def reset(self, B: int):
        """Start B episodes. Returns local state indices (B, N) in [0, 4)."""
        self._B = B
        # each interior slot gets a Bernoulli(0.5) packet at init (reference reset:130-131)
        self._state = (self.rng.random((B, self.N, self.ddl)) < 0.5).astype(np.int8)
        self._t = 0
        return self._state_index()

    def _state_index(self):
        return (self._state * self._bitw).sum(-1)  # (B, N) in [0,4)

    def step(self, actions: np.ndarray):
        """Advance one step.

        actions: (B, N) int in [0,5). Returns (state_idx, f, g, done) where
          f: (B, N) objective reward 1[successful uncontested transmission],
          g: (B, N) constraint reward -1[sampled action != Idle]  (brief section 5).
        """
        B, N = self._B, self.N
        st = self._state
        has_pkt = st.max(axis=-1) > 0  # (B, N)

        # constraint reward penalises the SAMPLED action (brief section 5 [SHOULD]).
        g = -(actions != IDLE).astype(np.float64)

        # effective target AP for dynamics (directional w/o AP -> remapped to Idle).
        target = self.topo.ap_target[np.arange(N)[None, :], actions]  # (B, N), -1 if none
        attempt = (target >= 0) & has_pkt  # transmissions that could succeed

        f = np.zeros((B, N), dtype=np.float64)
        bidx, aidx = np.nonzero(attempt)
        if bidx.size:
            tg = target[bidx, aidx]
            counts = np.zeros((B, self.topo.n_ap), dtype=np.int64)
            np.add.at(counts, (bidx, tg), 1)
            sole = counts[bidx, tg] == 1                      # uncontested at that AP
            qdraw = self.rng.random(bidx.size) <= self.q[tg]  # AP processes w.p. q_y
            success = sole & qdraw
            sb, sa, stg = bidx[success], aidx[success], None
            f[sb, sa] = 1.0
            # remove the earliest queued packet (most-left 1) on success (reference:212-219)
            earliest = st[sb, sa].argmax(axis=1)  # first slot with a 1
            st[sb, sa, earliest] = 0

        # deadline left-shift + append Bernoulli(p_i) arrivals (reference:222-225)
        st[:, :, :-1] = st[:, :, 1:]
        new = (self.rng.random((B, N)) < self.p[None, :]).astype(np.int8)
        st[:, :, -1] = new

        self._t += 1
        done = self._t >= self.H
        return self._state_index(), f, g, done

    def rollout(self, policy_sample, B: int):
        """Collect B full episodes under ``policy_sample``.

        policy_sample(state_idx) -> actions (B, N) int. Returns a dict of stacked arrays
        with shape (H, B, N): ``states`` (int idx), ``actions``, ``f`` (objective), ``g``
        (constraint). No discounting here; the metrics module applies gamma (brief section 6).
        """
        s = self.reset(B)
        H = self.H
        states = np.empty((H, B, self.N), dtype=np.int64)
        actions = np.empty((H, B, self.N), dtype=np.int64)
        fs = np.empty((H, B, self.N), dtype=np.float64)
        gs = np.empty((H, B, self.N), dtype=np.float64)
        for t in range(H):
            a = policy_sample(s)
            states[t] = s
            actions[t] = a
            s, f, g, done = self.step(a)
            fs[t] = f
            gs[t] = g
            if done:
                break
        return dict(states=states, actions=actions, f=fs, g=gs, H=H)
