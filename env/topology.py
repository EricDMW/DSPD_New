"""Topology of the 5x5 wireless access-control network.

Everything here is derived from brief section 4 and SPDAC Appendix H.3, verified against
the reference ``Decentralized-Safe-MARL-with-General-Utilities/envs/wireless_comm.py``
(``access_point_mapping``, agent ordering ``id = row*L + col``).

Two agents are neighbours iff they *share an access point* (brief section 4); this defines
the environmental graph ``G^E`` and its kappa-hop neighbourhoods ``N_i^{E,kappa}``.
"""

from __future__ import annotations

import numpy as np

# Action indexing (Idle = 0), matching wireless_comm.py:147-168.
IDLE, UL, LL, UR, LR = 0, 1, 2, 3, 4
ACTION_NAMES = ["Idle", "UL", "LL", "UR", "LR"]
N_ACTIONS = 5


def _ap_offsets(action):
    """Return the (drow, dcol) of the access point a directional action targets.

    Cell (i, j) sits at the centre of four AP corners. Mapping copied verbatim from the
    reference ``access_point_mapping`` (wireless_comm.py:150-161):
      a=1 UL -> (i-1, j-1), a=2 LL -> (i, j-1), a=3 UR -> (i-1, j), a=4 LR -> (i, j).
    """
    return {
        UL: (-1, -1),
        LL: (0, -1),
        UR: (-1, 0),
        LR: (0, 0),
    }[action]


class Topology:
    """Immutable topology for an ``L x L`` grid (L is fixed to 5 for all reported results)."""

    def __init__(self, L: int = 5):
        # brief section 4 / section 15: the environment is the 5x5 network, L is NOT a knob.
        assert L == 5, f"L must be 5 (the fixed 5x5 wireless network); got {L}"
        self.L = L
        self.N = L * L
        self.n_ap_x = L - 1
        self.n_ap_y = L - 1
        self.n_ap = self.n_ap_x * self.n_ap_y
        assert self.N == 25 and self.n_ap == 16, "must be N=25 agents, 16 access points"

        self._build_ap_map()
        self._build_adjacency()

    # -- access-point map -------------------------------------------------------------
    def _build_ap_map(self):
        """ap_target[agent, action] = flat AP id in [0, 16), or -1 if remapped to Idle."""
        L = self.L
        self.ap_target = -np.ones((self.N, N_ACTIONS), dtype=np.int64)
        self.agent_aps = [[] for _ in range(self.N)]  # Y_i: APs reachable by agent i
        for i in range(L):
            for j in range(L):
                agent = i * L + j
                for a in (UL, LL, UR, LR):
                    dr, dc = _ap_offsets(a)
                    x, y = i + dr, j + dc
                    if 0 <= x < self.n_ap_x and 0 <= y < self.n_ap_y:
                        ap = x * self.n_ap_y + y
                        self.ap_target[agent, a] = ap
                        self.agent_aps[agent].append(ap)
        # interior agents touch 4 APs, edge 2, corner 1 (brief section 4)
        degs = sorted({len(a) for a in self.agent_aps})
        assert degs == [1, 2, 4] or set(degs) <= {1, 2, 4}

    # -- shared-AP adjacency ----------------------------------------------------------
    def _build_adjacency(self):
        N = self.N
        ap_sets = [set(a) for a in self.agent_aps]
        A = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(N):
                if i != j and ap_sets[i] & ap_sets[j]:
                    A[i, j] = True
        self.adj = A
        # kappa-hop neighbourhoods (BFS on the shared-AP graph), including self.
        self._khop = {1: self._bfs_khops(1)}

    def _bfs_khops(self, kappa: int):
        """N_i^{E,kappa} for every i: agents within graph distance <= kappa (incl. self)."""
        N = self.N
        neigh = []
        for i in range(N):
            frontier = {i}
            seen = {i}
            for _ in range(kappa):
                nxt = set()
                for u in frontier:
                    nxt |= set(np.nonzero(self.adj[u])[0].tolist())
                nxt -= seen
                seen |= nxt
                frontier = nxt
            neigh.append(sorted(seen))
        return neigh

    def khop(self, kappa: int):
        """List (len N) of sorted kappa-hop neighbourhoods including self."""
        if kappa not in self._khop:
            self._khop[kappa] = self._bfs_khops(kappa)
        return self._khop[kappa]

    def khop_excl_self(self, kappa: int):
        """N_{i,-i}^{E,kappa}: kappa-hop neighbours excluding agent i (used in Eq. 49)."""
        return [[j for j in nb if j != i] for i, nb in enumerate(self.khop(kappa))]


class LearningNetwork:
    """Time-varying learning network ``{G^L_m}`` for the DSPD push-sum estimator.

    Draft Assumption 2 / Section II: a switching sequence of digraphs, uniformly strongly
    connected, *separate* from the environmental network ``G^E``. Column-stochastic weights
    per Eq. (1): ``w_{ij,m} = 1/|N^out_{j,m}|`` if i is in j's out-neighbourhood.

    Construction: a rotating directed Hamiltonian ring with self-loops. A directed ring is
    strongly connected on its own, so union-strong-connectivity holds over every window of
    length 1 (the requirement is asserted in :meth:`check_union_connectivity`). Rotating the
    ordering each step makes it genuinely time-varying (draft Section VI uses a small
    switching chain; a rotating ring is its N=25 generalisation).
    """

    def __init__(self, N: int, seed: int = 0, n_patterns: int = 3,
                 structure: str = "ring", extra_edge_p: float = 0.06):
        self.N = N
        self.structure = structure
        rng = np.random.default_rng(seed)
        # union_window D: the switching period over which the union of digraphs must be strongly
        # connected (Assumption 6). A single ring/random digraph is already strongly connected (D=1);
        # the two-structure SWITCHING network is NOT connected in either phase and needs D=2.
        self.union_window = 1
        if structure == "switch2":
            # Faithful port of the manuscript's numerical learning network, which "switches back and
            # forth between 1->2->3 and 3->4->1" (Sec. VI): a sequence of TWO digraphs, NEITHER
            # strongly connected on its own, whose UNION over the period is a single directed
            # Hamiltonian cycle -> uniformly strongly connected with D=2. Build one Hamiltonian cycle
            # order o (o[k] -> o[k+1]) and 2-colour its edges by parity: phase A carries the
            # even-indexed cycle edges, phase B the odd-indexed ones (each node also keeps a
            # self-loop). Neither phase reaches every node; their union is the whole cycle. Column-
            # stochastic weights w_{ij}=1/|out(j)| per Eq. (1); row-stochastic companion for the
            # estimation diagnostic.
            self.n_patterns = 2
            self.union_window = 2
            o = rng.permutation(N)
            self.orders = [o, o]                         # same cycle, two edge-parity phases
            self._weights, self._rowmats = [], []
            for parity in (0, 1):
                col, row = self._switch2_weights(o, parity)
                self._weights.append(col)
                self._rowmats.append(row)
        elif structure == "random":
            # Per-run RANDOM strongly-connected digraphs: a random Hamiltonian-cycle backbone
            # (guarantees strong connectivity) plus i.i.d. extra directed edges. Different runs
            # (seeds) get structurally different graphs -> different mixing rates -> the push-sum
            # estimation error genuinely varies across seeds (visible +/-std band). Still satisfies
            # Assumption 2 (uniform strong connectivity). Column- and row-stochastic weights are
            # uniform over each node's out-/in-neighbourhood.
            self.n_patterns = n_patterns
            self.orders = [rng.permutation(N) for _ in range(n_patterns)]
            self._weights, self._rowmats = [], []
            for o in self.orders:
                col, row = self._random_weights(o, rng, extra_edge_p)
                self._weights.append(col)
                self._rowmats.append(row)
        else:
            self.n_patterns = n_patterns
            self.orders = [rng.permutation(N) for _ in range(n_patterns)]
            self._weights = [self._ring_weights(o) for o in self.orders]
            self._rowmats = [self._ring_row(o) for o in self.orders]

    def _switch2_weights(self, order, parity):
        """Column- and row-stochastic matrices for one phase of the two-structure switching network.

        ``order`` is a Hamiltonian cycle o[0]->o[1]->...->o[N-1]->o[0]. Edge e_k = (o[k]->o[(k+1)%N])
        belongs to this phase iff k % 2 == parity. A node that is the tail of an active edge splits
        its column mass 1/2 self, 1/2 to its cycle successor; every other node is a pure self-loop.
        The two phases (parity 0 and 1) together carry all N cycle edges."""
        N = self.N
        W = np.zeros((N, N))   # column-stochastic push-sum weights (Eq. 1)
        A = np.zeros((N, N))   # row-stochastic companion (consensus diagnostic)
        active_out = np.zeros(N, dtype=bool)
        active_in = np.zeros(N, dtype=bool)
        for k in range(N):
            if k % 2 != parity:
                continue
            j = order[k]
            nxt = order[(k + 1) % N]
            W[j, j] += 0.5
            W[nxt, j] += 0.5
            active_out[j] = True
            A[nxt, j] += 0.5      # nxt averages in j's value
            active_in[nxt] = True
        for v in range(N):
            if not active_out[v]:
                W[v, v] = 1.0     # pure self-loop column
            if not active_in[v]:
                A[v, v] = 1.0     # pure self-loop row
            else:
                A[v, v] += 0.5
        return W, A

    def _ring_weights(self, order):
        """Column-stochastic W for a directed ring order[k] -> order[k+1] plus self-loops."""
        N = self.N
        W = np.zeros((N, N))
        for k in range(N):
            j = order[k]
            nxt = order[(k + 1) % N]
            # out-neighbourhood of j is {j (self), nxt}; column j sums to 1 (Eq. 1).
            W[j, j] += 0.5
            W[nxt, j] += 0.5
        return W

    def _ring_row(self, order):
        """Row-stochastic A: each node averages its own value with its ring predecessor's."""
        N = self.N
        A = np.zeros((N, N))
        for idx in range(N):
            k = order[idx]
            pred = order[(idx - 1) % N]
            A[k, k] += 0.5
            A[k, pred] += 0.5
        return A

    def _random_weights(self, order, rng, p):
        """Random strongly-connected digraph (Hamiltonian backbone + extra edges) -> (W_col, A_row).

        ``In[i]`` is agent i's in-neighbourhood (incl. self and its ring predecessor). Extra edges
        j->i are added i.i.d. with prob ``p``. Row-stochastic A[i,j]=1/|In[i]|; column-stochastic
        W[i,j]=1/|Out[j]|. Structurally distinct per seed, so the mixing rate varies across runs.
        """
        N = self.N
        In = [{i} for i in range(N)]                 # self-loops
        for idx in range(N):                          # Hamiltonian backbone pred -> node
            k, pred = order[idx], order[(idx - 1) % N]
            In[k].add(pred)
        extra = rng.random((N, N)) < p
        for i in range(N):
            for j in range(N):
                if i != j and extra[i, j]:
                    In[i].add(j)                      # edge j -> i
        A = np.zeros((N, N))                           # row-stochastic (consensus averaging)
        for i in range(N):
            for j in In[i]:
                A[i, j] = 1.0 / len(In[i])
        W = np.zeros((N, N))                           # column-stochastic (push-sum weights, Eq.1)
        for j in range(N):
            out = [i for i in range(N) if j in In[i]]
            for i in out:
                W[i, j] = 1.0 / len(out)
        return W, A

    def weight(self, m: int):
        """Column-stochastic weight matrix W_m at iteration m (1-indexed), per Eq. (1)."""
        return self._weights[(m - 1) % self.n_patterns]

    def row_stochastic(self, m: int):
        """Row-stochastic mixing matrix for consensus averaging at iteration m; with a clamped
        source it drives every estimate to the source value (push-sum theta_hat^i_j -> theta_j)."""
        return self._rowmats[(m - 1) % self.n_patterns]

    def check_union_connectivity(self, window: int = 1) -> bool:
        """Assert the union of any ``window`` consecutive digraphs is strongly connected."""
        for start in range(self.n_patterns):
            U = np.zeros((self.N, self.N), dtype=bool)
            for d in range(window):
                W = self._weights[(start + d) % self.n_patterns]
                U |= (W.T > 0)  # edge j->i where W[i,j]>0
            if not _strongly_connected(U):
                return False
        return True


def _strongly_connected(adj_bool) -> bool:
    """True iff the directed graph with boolean adjacency (edge i->j = adj[i,j]) is SCC."""
    N = adj_bool.shape[0]

    def reach(a):
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in np.nonzero(a[u])[0]:
                if v not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        return len(seen) == N

    return reach(adj_bool) and reach(adj_bool.T)
