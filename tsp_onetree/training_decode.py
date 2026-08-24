from typing import Dict, List, Tuple
import atexit
import concurrent.futures as cf
import math
import multiprocessing as mp
import time

import numpy as np
import torch

def prim_minimum_spanning_tree(cost: np.ndarray, root: int = 0) -> np.ndarray:
    """Minimum spanning tree adjacency from a dense symmetric cost matrix."""
    c = np.asarray(cost, dtype=np.float64)
    n = c.shape[0]
    root = int(root)
    in_tree = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=np.int64)
    best = np.full(n, np.inf, dtype=np.float64)
    best[root] = 0.0

    for _ in range(n):
        masked_best = np.where(in_tree, np.inf, best)
        u = int(np.argmin(masked_best))
        in_tree[u] = True
        better = (~in_tree) & (c[u] < best)
        parent[better] = u
        best = np.where(better, c[u], best)

    adj = np.zeros((n, n), dtype=np.int64)
    rows = np.where(parent >= 0)[0]
    for v in rows.tolist():
        u = int(parent[v])
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def cycle_from_degree2_edges(edges: List[Tuple[int, int]], n: int, root: int = 0) -> list[int] | None:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    if any(len(nei) != 2 for nei in adj):
        return None
    tour = [root]
    prev = -1
    cur = root
    for _ in range(n - 1):
        nxts = adj[cur]
        nxt = nxts[0] if nxts[0] != prev else nxts[1]
        if nxt == root:
            return None
        tour.append(nxt)
        prev, cur = cur, nxt
    if root not in adj[cur]:
        return None
    if len(set(tour)) != n:
        return None
    return tour



def _tour_cost_numpy(tour: list[int], D: np.ndarray) -> float:
    idx = np.asarray(tour, dtype=np.int64)
    nxt = np.asarray(tour[1:] + tour[:1], dtype=np.int64)
    return float(np.sum(D[idx, nxt]))


def _sym_mu_matching_cost(mu: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu_arr = np.asarray(mu, dtype=np.float64)
    mu_sym = 0.5 * (mu_arr + mu_arr.T)
    return -np.log(np.clip(mu_sym, eps, None))


def _sym_cmod_matching_cost(C_mod_single: np.ndarray) -> np.ndarray:
    C = np.asarray(C_mod_single, dtype=np.float64)
    return 0.5 * (C + C.T)


def _sym_metric_matching_cost(D_single: np.ndarray) -> np.ndarray:
    D = np.asarray(D_single, dtype=np.float64)
    return 0.5 * (D + D.T)



def two_opt_rooted_tour_preserve_root_endpoints(
    tour: list[int],
    D: np.ndarray,
    max_passes: int = 6,
) -> list[int]:
    """2-opt on a rooted tour [root, a, ..., b] while preserving root neighbors a and b."""
    n = len(tour)
    if n <= 5 or max_passes <= 0:
        return tour
    arr = np.asarray(tour, dtype=np.int64).copy()
    passes = 0
    while passes < max_passes:
        passes += 1
        improved = False
        for i in range(2, n - 2):
            a = int(arr[i - 1])
            b = int(arr[i])
            for j in range(i + 1, n - 1):
                c = int(arr[j])
                d = int(arr[(j + 1) % n])
                delta = float(D[a, c] + D[b, d] - D[a, b] - D[c, d])
                if delta < -1e-12:
                    arr[i : j + 1] = arr[i : j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr.tolist()



def _adjacency_lists(adj: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(adj[u]).astype(int).tolist() for u in range(adj.shape[0])]



def _tree_path(adj: np.ndarray, start: int, goal: int) -> list[int]:
    """Unique path between start and goal in a tree adjacency matrix."""
    n = adj.shape[0]
    start = int(start)
    goal = int(goal)
    parent = np.full(n, -2, dtype=np.int64)
    stack = [start]
    parent[start] = -1
    while stack:
        u = int(stack.pop())
        if u == goal:
            break
        for v in np.flatnonzero(adj[u]).astype(int).tolist():
            if parent[v] != -2:
                continue
            parent[v] = u
            stack.append(v)
    if parent[goal] == -2:
        raise RuntimeError(f"Tree path from {start} to {goal} not found.")
    path = [goal]
    cur = goal
    while parent[cur] != -1:
        cur = int(parent[cur])
        path.append(cur)
    path.reverse()
    return path



def _subtree_preorder_excluding_backbone(
    adj: np.ndarray,
    C_mod: np.ndarray,
    node: int,
    parent: int,
    backbone_set: set[int],
) -> list[int]:
    """Iterative preorder, preserving the former recursive DFS order.

    Deep MAP trees can contain paths longer than Python's recursion limit.  A
    stack avoids that size-dependent failure.  Children are pushed in reverse
    sorted order so popping visits them in exactly the same order as the former
    recursive implementation.
    """
    order: list[int] = []
    stack: list[tuple[int, int]] = [(int(node), int(parent))]
    while stack:
        current, current_parent = stack.pop()
        order.append(current)
        children = [
            int(v)
            for v in np.flatnonzero(adj[current]).astype(int).tolist()
            if int(v) != current_parent and int(v) not in backbone_set
        ]
        children.sort(key=lambda v: (float(C_mod[current, v]), int(v)))
        stack.extend((child, current) for child in reversed(children))
    return order



def _off_backbone_preorder(
    adj: np.ndarray,
    C_mod: np.ndarray,
    backbone: list[int],
) -> list[int]:
    """Deterministic order of off-backbone nodes guided by the MAP tree and C_mod."""
    backbone_set = set(int(v) for v in backbone)
    order: list[int] = []
    for u in backbone:
        children = [
            int(v)
            for v in np.flatnonzero(adj[u]).astype(int).tolist()
            if int(v) not in backbone_set
        ]
        children.sort(key=lambda v: (float(C_mod[u, v]), int(v)))
        for child in children:
            order.extend(_subtree_preorder_excluding_backbone(adj, C_mod, child, u, backbone_set))
    return order



def _insert_node_cheapest_path_position(path: list[int], node: int, C_mod: np.ndarray) -> None:
    """Insert node into the rooted nonroot path while preserving the endpoints."""
    if len(path) < 2:
        path.append(int(node))
        return
    best_pos = 1
    best_delta = float('inf')
    node = int(node)
    for i in range(len(path) - 1):
        u = int(path[i])
        v = int(path[i + 1])
        delta = float(C_mod[u, node] + C_mod[node, v] - C_mod[u, v])
        cand = (delta, float(C_mod[u, node] + C_mod[node, v]), i + 1)
        if cand < (best_delta, float('inf'), best_pos):
            best_delta = cand[0]
            best_pos = cand[2]
    path.insert(best_pos, node)



def map_rooted_onetree_from_cmod(
    C_mod_single: np.ndarray,
    root: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], np.ndarray]:
    """Compute the MAP rooted 1-tree under reduced costs C_mod."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    root = int(root)
    nonroot = [v for v in range(n) if v != root]
    if len(nonroot) < 2:
        raise ValueError("Need at least two non-root nodes for rooted 1-tree decoding.")

    C_nr = C[np.ix_(nonroot, nonroot)]
    adj_nr = prim_minimum_spanning_tree(C_nr, root=0)

    root_costs = np.asarray(C[root, nonroot], dtype=np.float64)
    order = np.argsort(root_costs, kind='mergesort')
    a = int(nonroot[int(order[0])])
    b = int(nonroot[int(order[1])])

    full_adj = np.zeros((n, n), dtype=np.int64)
    rows, cols = np.where(np.triu(adj_nr, k=1) > 0)
    edges: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        u = int(nonroot[i])
        v = int(nonroot[j])
        full_adj[u, v] = 1
        full_adj[v, u] = 1
        edges.append((u, v))
    full_adj[root, a] = 1
    full_adj[a, root] = 1
    full_adj[root, b] = 1
    full_adj[b, root] = 1
    edges.append((root, a))
    edges.append((root, b))
    return full_adj, edges, (a, b), np.asarray(nonroot, dtype=np.int64)



def _tree_hop_distance(adj: np.ndarray, start: int) -> np.ndarray:
    """Hop distances from start in a tree adjacency matrix."""
    n = adj.shape[0]
    start = int(start)
    dist = np.full(n, n + 1, dtype=np.int64)
    dist[start] = 0
    q = [start]
    head = 0
    while head < len(q):
        u = int(q[head])
        head += 1
        for v in np.flatnonzero(adj[u]).astype(int).tolist():
            if dist[v] <= dist[u] + 1:
                continue
            dist[v] = dist[u] + 1
            q.append(int(v))
    return dist



def backbone_insertion_path_from_rooted_onetree(
    full_adj: np.ndarray,
    C_mod_single: np.ndarray,
    root: int,
    root_pair: tuple[int, int],
) -> list[int]:
    """Construct a non-root Hamiltonian path by preserving the MAP-tree backbone.

    The path starts with the unique tree path between the two MAP root neighbors and then
    inserts off-backbone nodes in a deterministic tree-aware order by cheapest insertion
    under C_mod. This is less myopic than single-ended sequential growth.
    """
    n = full_adj.shape[0]
    root = int(root)
    a, b = int(root_pair[0]), int(root_pair[1])
    nonroot = [int(v) for v in range(n) if int(v) != root]
    idx_of = {int(v): i for i, v in enumerate(nonroot)}
    adj_nr = full_adj[np.ix_(nonroot, nonroot)]
    a_nr = idx_of[a]
    b_nr = idx_of[b]
    backbone_nr = _tree_path(adj_nr, a_nr, b_nr)
    backbone = [int(nonroot[idx]) for idx in backbone_nr]
    path = backbone.copy()
    off_nodes = _off_backbone_preorder(adj_nr, C_mod_single[np.ix_(nonroot, nonroot)], backbone_nr)
    for node_nr in off_nodes:
        node = int(nonroot[int(node_nr)])
        if node in path:
            continue
        _insert_node_cheapest_path_position(path, node, C_mod_single)
    return path



def _tree_edge_priority_for_removal(
    full_adj: np.ndarray,
    u: int,
    mu: np.ndarray,
    root: int,
    root_pair: tuple[int, int],
) -> list[tuple[int, float]]:
    """For vertex u with tree degree > 2 in the rooted 1-tree, enumerate which of
    its incident tree edges are 'safe' to remove (i.e., not a root edge protected
    by the MAP root pair), and return (neighbor, removal_priority) pairs.

    Priority is -log(mu_e + eps): low priority (small mu) = preferred removal.
    """
    EPS = 1e-8
    n = full_adj.shape[0]
    out: list[tuple[int, float]] = []
    protected = {(int(root), int(root_pair[0])), (int(root_pair[0]), int(root)),
                 (int(root), int(root_pair[1])), (int(root_pair[1]), int(root))}
    for v in np.flatnonzero(full_adj[u]).tolist():
        v = int(v)
        if v == u:
            continue
        if (u, v) in protected:
            continue  # Do not remove a protected root edge
        m = float(mu[u, v])
        # Priority: -log(mu+eps) — LOW mu → LARGE priority (good candidate to remove)
        priority = -math.log(m + EPS)
        out.append((v, priority))
    # Sort by priority descending (prefer edges with lowest mu to remove first)
    out.sort(key=lambda pv: -pv[1])
    return out


def mu_weighted_matching_repair(
    full_adj: np.ndarray,
    mu: np.ndarray,
    C_mod_single: np.ndarray,
    D_single: np.ndarray,
    root: int,
    root_pair: tuple[int, int],
    twoopt_passes: int = 0,
) -> tuple[list[int] | None, bool]:
    """Prior-faithful repair: use the model's learned marginals μ to score swaps,
    rather than C_mod cost. Solves a minimum-cost matching on the degree-violation set,
    where swap cost = -log(μ_added) + log(μ_removed), i.e. 'log-likelihood decrease
    under the learned 1-tree measure'.

    Returns (tour, success). If success=False, caller falls back to the existing
    backbone-insertion repair.

    Handles:
      - k=1 (one deg-1 + one deg-3 pair): optimal swap by argmin
      - k=2, 3: brute-force enumeration of small matchings
      - k>=4: scipy Hungarian (if available)

    Connectivity correctness: after swaps, we must have n edges forming a
    Hamiltonian cycle. We check this explicitly and return failure if not.
    """
    n = full_adj.shape[0]
    root = int(root)
    deg = full_adj.sum(axis=1)

    # Identify deg-1 (deficit, need a new edge) and deg-3+ (excess, need to drop one)
    # among non-root vertices. Root always has degree 2 in a rooted 1-tree MAP.
    nonroot_mask = np.ones(n, dtype=bool)
    nonroot_mask[root] = False
    deg1_vertices = [int(v) for v in range(n) if nonroot_mask[v] and deg[v] == 1]
    deg3_vertices = [int(v) for v in range(n) if nonroot_mask[v] and deg[v] >= 3]

    # Guard: require balanced counts. Each deg-3 has excess-1, each deg-1 has deficit-1.
    # If deg-4 or worse appears, this simplified matcher doesn't handle it —
    # fall back.
    if len(deg1_vertices) != len(deg3_vertices):
        return None, False
    if any(deg[u] > 3 for u in deg3_vertices):
        return None, False
    k = len(deg1_vertices)
    if k == 0:
        # Already a tour — shouldn't be called, but handle gracefully
        return None, False
    # Cap problem size; beyond this, fall back to insertion repair.
    if k > 6:
        return None, False

    EPS = 1e-8
    mu_sym = 0.5 * (mu + mu.T)

    # Build the cost matrix for the assignment problem.
    # Row i = deg-3 vertex u_i, col j = deg-1 vertex v_j.
    # SWAP: remove edge (u_i, x) at u_i, and add edge (x, v_j) where x is
    # the orphaned tree neighbor. This properly rebalances degrees:
    #   u_i : deg 3 → 2  (lost one neighbor x)
    #   x   : deg d → d  (lost u_i but gained v_j, unchanged)
    #   v_j : deg 1 → 2  (gained x)
    # Score = -log(mu_{x, v_j} + eps) - (-log(mu_{u_i, x} + eps))
    #       = -log(mu_{x, v_j}) + log(mu_{u_i, x})
    # Low cost = high-probability add AND low-probability remove.
    # We pick the best (x, mu_{u,x}) pair for each (u, v).
    cost_matrix = np.full((k, k), np.inf, dtype=np.float64)
    removal_choice = np.zeros((k, k), dtype=np.int64)  # which x (= neighbor of u_i to remove)
    for i, u in enumerate(deg3_vertices):
        removal_options = _tree_edge_priority_for_removal(full_adj, u, mu_sym, root, root_pair)
        if len(removal_options) == 0:
            continue
        for j, v in enumerate(deg1_vertices):
            if u == v:
                continue
            # Try each removal candidate x and take the best combined swap score.
            best_swap_cost = np.inf
            best_x = -1
            for (x, remove_priority) in removal_options:
                if x == v:
                    continue  # degenerate
                if full_adj[x, v] > 0:
                    continue  # edge (x,v) already exists — swap would not change adjacency,
                              # leaving v with degree 1 and x with degree d-1 (bad)
                add_score = -math.log(float(mu_sym[x, v]) + EPS)
                swap_cost = add_score - remove_priority
                if swap_cost < best_swap_cost:
                    best_swap_cost = swap_cost
                    best_x = x
            if best_x >= 0 and np.isfinite(best_swap_cost):
                cost_matrix[i, j] = best_swap_cost
                removal_choice[i, j] = best_x

    if not np.isfinite(cost_matrix).any():
        return None, False

    # Solve assignment
    if k == 1:
        assignment = np.array([0], dtype=np.int64)
    elif k <= 3:
        # Brute force: enumerate all k! permutations
        from itertools import permutations
        best_cost = np.inf
        best_perm = None
        for perm in permutations(range(k)):
            c = sum(cost_matrix[i, perm[i]] for i in range(k))
            if c < best_cost:
                best_cost = c
                best_perm = perm
        if best_perm is None or not np.isfinite(best_cost):
            return None, False
        assignment = np.array(best_perm, dtype=np.int64)
    else:
        # Hungarian via scipy
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignment = col_ind
        except Exception:
            return None, False
        if not np.all(np.isfinite(cost_matrix[np.arange(k), assignment])):
            return None, False

    # Apply the swaps
    new_adj = full_adj.copy()
    for i in range(k):
        u = deg3_vertices[i]
        v = deg1_vertices[assignment[i]]
        x = int(removal_choice[i, assignment[i]])  # orphan endpoint
        # Remove edge (u, x)
        new_adj[u, x] = 0
        new_adj[x, u] = 0
        # Add edge (x, v): reconnect orphan to deg-1 vertex
        new_adj[x, v] = 1
        new_adj[v, x] = 1

    # Verify: must now be degree-2 everywhere, connected, one cycle
    new_deg = new_adj.sum(axis=1)
    if not np.all(new_deg == 2):
        return None, False

    edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if new_adj[i, j] > 0]
    if len(edges) != n:
        return None, False

    # Extract cycle from edges
    tour = cycle_from_degree2_edges(edges, n=n, root=root)
    if tour is None:
        return None, False

    if twoopt_passes > 0:
        tour = two_opt_rooted_tour_preserve_root_endpoints(
            tour, D_single, max_passes=max(1, min(2, int(twoopt_passes)))
        )
    return tour, True


def christofides_like_repair(
    full_adj: np.ndarray,
    matching_cost: np.ndarray,
    D_single: np.ndarray,
    root: int,
    twoopt_passes: int = 0,
) -> tuple[list[int] | None, bool]:
    """Eulerize a rooted 1-tree by matching odd-degree vertices, then shortcut."""
    try:
        import networkx as nx
    except Exception:
        return None, False

    adj = np.asarray(full_adj, dtype=np.int64)
    n = int(adj.shape[0])
    root = int(root)
    deg = adj.sum(axis=1)
    odd_vertices = [int(v) for v in range(n) if int(deg[v]) % 2 == 1]
    if len(odd_vertices) % 2 != 0:
        return None, False

    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] > 0:
                multigraph.add_edge(int(u), int(v))

    if odd_vertices:
        match_graph = nx.Graph()
        match_graph.add_nodes_from(odd_vertices)
        for i, u in enumerate(odd_vertices):
            for v in odd_vertices[i + 1 :]:
                w = float(matching_cost[u, v])
                if not np.isfinite(w):
                    return None, False
                match_graph.add_edge(int(u), int(v), weight=w)
        try:
            matching = nx.algorithms.matching.min_weight_matching(match_graph, weight="weight")
        except Exception:
            return None, False
        if 2 * len(matching) != len(odd_vertices):
            return None, False
        for u, v in matching:
            multigraph.add_edge(int(u), int(v))

    try:
        walk = [root]
        for edge in nx.eulerian_circuit(multigraph, source=root):
            _, v = edge[:2]
            walk.append(int(v))
    except Exception:
        return None, False

    seen: set[int] = set()
    tour: list[int] = []
    for node in walk:
        node = int(node)
        if node in seen:
            continue
        seen.add(node)
        tour.append(node)
    if len(tour) != n:
        return None, False
    tour = _rotate_tour_to_root(tour, root)
    if twoopt_passes > 0:
        tour = two_opt_rooted_tour_preserve_root_endpoints(
            tour, D_single, max_passes=max(1, min(2, int(twoopt_passes)))
        )
    return tour, True


def repair_rooted_onetree_to_tour(
    full_adj: np.ndarray,
    C_mod_single: np.ndarray,
    D_single: np.ndarray,
    root: int,
    root_pair: tuple[int, int],
    twoopt_passes: int = 0,
    mu_single: np.ndarray | None = None,
    repair_mode: str = "auto",
) -> tuple[list[int], bool]:
    """Deterministic tree-aware repair from a rooted 1-tree to a valid tour.

    Returns (tour, used_repair). Path:
      1. If the MAP rooted 1-tree is already degree-2 everywhere, extract cycle directly.
      2. Else try the configured repair mode:
         - auto: preserve historical behavior (μ-match if available, else insertion)
         - mu_match: local μ-weighted swap repair
         - mu_christofides: parity matching with -log(μ) edge weights
         - cmod_christofides: parity matching with C_mod edge weights
         - c_christofides: parity matching with original metric D edge weights
         - backbone_insert: skip matching and use insertion directly
      3. If the configured repair fails, fall back to backbone-insertion repair.
    """
    n = full_adj.shape[0]
    root = int(root)
    edges = [(int(i), int(j)) for i in range(n) for j in range(i + 1, n) if full_adj[i, j] > 0]
    deg = full_adj.sum(axis=1)
    if np.all(deg == 2):
        exact_tour = cycle_from_degree2_edges(edges, n=n, root=root)
        if exact_tour is not None:
            exact_tour = two_opt_rooted_tour_preserve_root_endpoints(exact_tour, D_single, max_passes=max(0, min(2, int(twoopt_passes))))
            return exact_tour, False

    mode = str(repair_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "mu_match" if mu_single is not None else "backbone_insert"

    if mode == "mu_match" and mu_single is not None:
        matched_tour, ok = mu_weighted_matching_repair(
            full_adj, mu_single, C_mod_single, D_single,
            root=root, root_pair=root_pair, twoopt_passes=twoopt_passes,
        )
        if ok and matched_tour is not None:
            return matched_tour, True
    elif mode == "mu_christofides" and mu_single is not None:
        matched_tour, ok = christofides_like_repair(
            full_adj,
            matching_cost=_sym_mu_matching_cost(mu_single),
            D_single=D_single,
            root=root,
            twoopt_passes=twoopt_passes,
        )
        if ok and matched_tour is not None:
            return matched_tour, True
    elif mode == "cmod_christofides":
        matched_tour, ok = christofides_like_repair(
            full_adj,
            matching_cost=_sym_cmod_matching_cost(C_mod_single),
            D_single=D_single,
            root=root,
            twoopt_passes=twoopt_passes,
        )
        if ok and matched_tour is not None:
            return matched_tour, True
    elif mode == "c_christofides":
        matched_tour, ok = christofides_like_repair(
            full_adj,
            matching_cost=_sym_metric_matching_cost(D_single),
            D_single=D_single,
            root=root,
            twoopt_passes=twoopt_passes,
        )
        if ok and matched_tour is not None:
            return matched_tour, True

    # Fallback: backbone + cheapest-insertion repair (C_mod based)
    path = backbone_insertion_path_from_rooted_onetree(
        full_adj,
        C_mod_single,
        root=root,
        root_pair=root_pair,
    )
    tour = [root] + path
    if twoopt_passes > 0:
        tour = two_opt_rooted_tour_preserve_root_endpoints(tour, D_single, max_passes=max(1, min(2, int(twoopt_passes))))
    return tour, True



def two_opt_cycle(tour: list[int], D: np.ndarray, max_passes: int = 6) -> list[int]:
    """Standard 2-opt on a cycle represented as an ordered tour list."""
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = np.asarray(tour, dtype=np.int64).copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = int(arr[i])
            b = int(arr[(i + 1) % n])
            for j in range(i + 2, n if i > 0 else n - 1):
                c = int(arr[j])
                d = int(arr[(j + 1) % n])
                delta = float(D[a, c] + D[b, d] - D[a, b] - D[c, d])
                if delta < -1e-12:
                    arr[i + 1 : j + 1] = arr[i + 1 : j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr.tolist()



def _rotate_tour_to_root(tour: list[int], root: int) -> list[int]:
    root = int(root)
    if tour[0] == root:
        return tour
    ridx = tour.index(root)
    return tour[ridx:] + tour[:ridx]



def greedy_degree2_tour_from_scores(score: np.ndarray, D: np.ndarray, root: int = 0) -> tuple[list[int], bool]:
    """Greedy Hamiltonian cycle extraction from a dense symmetric edge score matrix."""
    n = score.shape[0]
    deg = np.zeros(n, dtype=int)
    dsu = _DSU(n)
    selected: list[tuple[int, int]] = []
    all_edges: list[tuple[float, float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((float(score[i, j]), -float(D[i, j]), i, j))
    all_edges.sort(reverse=True)

    for _, _, u, v in all_edges:
        if deg[u] >= 2 or deg[v] >= 2:
            continue
        same = dsu.find(u) == dsu.find(v)
        if same:
            if len(selected) == n - 1 and deg[u] == 1 and deg[v] == 1:
                selected.append((u, v))
                deg[u] += 1
                deg[v] += 1
                break
            continue
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    if len(selected) == n:
        tour = cycle_from_degree2_edges(selected, n=n, root=root)
        if tour is not None:
            return tour, False

    # Very rare fallback: nearest-neighbor style completion on scores.
    root = int(root)
    vis = {root}
    order = [root]
    cur = root
    while len(order) < n:
        best = None
        best_v = None
        for v in range(n):
            if v in vis:
                continue
            key = (-float(score[cur, v]), float(D[cur, v]), int(v))
            if best is None or key < best:
                best = key
                best_v = int(v)
        order.append(best_v)
        vis.add(best_v)
        cur = best_v
    return order, True



def _topk_indices_desc(values: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
    arr = np.asarray(values)
    n = arr.shape[0]
    if exclude is not None:
        arr = arr.copy()
        arr[int(exclude)] = -np.inf
    k = max(0, min(int(k), n - (1 if exclude is not None else 0)))
    if k <= 0:
        return []
    idx = np.argpartition(-arr, kth=np.arange(k))[:k]
    idx = idx[np.argsort(-arr[idx], kind='mergesort')]
    return [int(i) for i in idx.tolist() if (exclude is None or int(i) != int(exclude))]



def _topk_indices_asc(values: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
    arr = np.asarray(values)
    n = arr.shape[0]
    if exclude is not None:
        arr = arr.copy()
        arr[int(exclude)] = np.inf
    k = max(0, min(int(k), n - (1 if exclude is not None else 0)))
    if k <= 0:
        return []
    idx = np.argpartition(arr, kth=np.arange(k))[:k]
    idx = idx[np.argsort(arr[idx], kind='mergesort')]
    return [int(i) for i in idx.tolist() if (exclude is None or int(i) != int(exclude))]



def build_move_candidates(
    mu_single: np.ndarray,
    C_mod_single: np.ndarray,
    cand_mask_single: np.ndarray | None,
    extra_topk: int = 5,
) -> list[set[int]]:
    """Candidate edge set per node for local search.

    Union of geometric candidates, top-μ support, and lowest-C_mod neighbors.
    """
    n = C_mod_single.shape[0]
    mu_sym = 0.5 * (mu_single + mu_single.T)
    C_sym = 0.5 * (C_mod_single + C_mod_single.T)
    cands: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        if cand_mask_single is not None:
            for j in np.flatnonzero(cand_mask_single[i]).astype(int).tolist():
                if int(j) != i:
                    cands[i].add(int(j))
        for j in _topk_indices_desc(mu_sym[i], extra_topk, exclude=i):
            cands[i].add(int(j))
        for j in _topk_indices_asc(C_sym[i], extra_topk, exclude=i):
            cands[i].add(int(j))
    for i in range(n):
        for j in list(cands[i]):
            cands[j].add(i)
        cands[i].discard(i)
    return cands



def candidate_restricted_two_opt(
    tour: list[int],
    D: np.ndarray,
    candidate_sets: list[set[int]],
    max_passes: int = 6,
    C_mod: np.ndarray | None = None,
) -> list[int]:
    """Candidate-restricted 2-opt on a cycle.

    If C_mod is provided, candidates for each anchor edge (a,b) are tried in ascending
    order of C_mod[a, c] — this is the LK-alpha ordering that biases first-improvement
    search along the learned Lagrangian gradient. The acceptance criterion is unchanged
    (real distance D), so terminal local optima are correct regardless of ordering.
    """
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = tour.copy()
    pos = {int(v): i for i, v in enumerate(arr)}
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            a = int(arr[i])
            b = int(arr[(i + 1) % n])
            cand_union = set(candidate_sets[a]) | set(candidate_sets[b])
            if C_mod is not None:
                # Rank candidates ascending by C_mod[a, c] — smallest reduced cost first.
                cand_nodes = sorted(cand_union, key=lambda c: (float(C_mod[a, int(c)]), int(c)))
            else:
                cand_nodes = cand_union
            for c in cand_nodes:
                j = pos.get(int(c), -1)
                if j < 0:
                    continue
                if j == i or j == (i + 1) % n or (j + 1) % n == i:
                    continue
                if i < j:
                    i1, j1 = i, j
                else:
                    i1, j1 = j, i
                a1 = int(arr[i1])
                b1 = int(arr[(i1 + 1) % n])
                c1 = int(arr[j1])
                d1 = int(arr[(j1 + 1) % n])
                if b1 == c1 or d1 == a1:
                    continue
                delta = float(D[a1, c1] + D[b1, d1] - D[a1, b1] - D[c1, d1])
                if delta < -1e-12:
                    arr[i1 + 1 : j1 + 1] = arr[i1 + 1 : j1 + 1][::-1]
                    pos = {int(v): k for k, v in enumerate(arr)}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr



def candidate_restricted_relocate_one(
    tour: list[int],
    D: np.ndarray,
    candidate_sets: list[set[int]],
    max_passes: int = 4,
    C_mod: np.ndarray | None = None,
) -> list[int]:
    """Candidate-restricted relocate-one on a cycle.

    If C_mod is provided, reinsertion candidates y are tried in ascending order of
    C_mod[x, y] — smallest reduced cost first. Acceptance still uses D.
    """
    n = len(tour)
    if n <= 4 or max_passes <= 0:
        return tour
    arr = tour.copy()
    for _ in range(max_passes):
        pos = {int(v): i for i, v in enumerate(arr)}
        improved = False
        for i in range(n):
            x = int(arr[i])
            prev_x = int(arr[(i - 1) % n])
            next_x = int(arr[(i + 1) % n])
            remove_gain = float(D[prev_x, next_x] - D[prev_x, x] - D[x, next_x])
            cand_iter = candidate_sets[x]
            if C_mod is not None:
                cand_iter = sorted(cand_iter, key=lambda y: (float(C_mod[x, int(y)]), int(y)))
            for y in cand_iter:
                j = pos.get(int(y), -1)
                if j < 0:
                    continue
                if j == i or (j + 1) % n == i or j == (i - 1) % n:
                    continue
                y_next = int(arr[(j + 1) % n])
                delta = remove_gain + float(D[y, x] + D[x, y_next] - D[y, y_next])
                if delta < -1e-12:
                    node = arr.pop(i)
                    if i < j:
                        j -= 1
                    arr.insert(j + 1, node)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return arr



def lk_lite_improve_tour(
    seed_tour: list[int],
    D: np.ndarray,
    candidate_sets: list[set[int]],
    twoopt_passes: int = 6,
    C_mod: np.ndarray | None = None,
) -> list[int]:
    """Lightweight LKH-style improvement: candidate-restricted 2-opt + relocate-1.

    If C_mod is provided, inner candidate orderings use ascending C_mod (LK-alpha).
    The final plain 2-opt cleanup is unchanged (unrestricted, uses D only).
    """
    tour = seed_tour.copy()
    outer = max(1, int(twoopt_passes)) if twoopt_passes > 0 else 0
    if outer <= 0:
        return tour
    for _ in range(outer):
        before = _tour_cost_numpy(tour, D)
        tour = candidate_restricted_two_opt(tour, D, candidate_sets, max_passes=max(1, twoopt_passes), C_mod=C_mod)
        tour = candidate_restricted_relocate_one(tour, D, candidate_sets, max_passes=max(1, twoopt_passes // 2 + 1), C_mod=C_mod)
        after = _tour_cost_numpy(tour, D)
        if after >= before - 1e-12:
            break
    tour = two_opt_cycle(tour, D, max_passes=max(1, twoopt_passes))
    return tour



def _mu_score_matrix(mu_single: np.ndarray) -> np.ndarray:
    mu_sym = 0.5 * (mu_single + mu_single.T)
    score = np.log(np.clip(mu_sym, 1e-12, None))
    np.fill_diagonal(score, -np.inf)
    return score



def mu_greedy_seed(mu_single: np.ndarray, D_single: np.ndarray, root: int = 0) -> tuple[list[int], bool]:
    score = _mu_score_matrix(mu_single)
    return greedy_degree2_tour_from_scores(score, D_single, root=root)



def mu_gumbel_seed(
    mu_single: np.ndarray,
    D_single: np.ndarray,
    rng: np.random.Generator,
    root: int = 0,
    gumbel_scale: float = 0.20,
) -> tuple[list[int], bool]:
    score = _mu_score_matrix(mu_single)
    if gumbel_scale > 0.0:
        score = score + float(gumbel_scale) * _symmetric_gumbel_noise(score.shape[0], rng)
        np.fill_diagonal(score, -np.inf)
    return greedy_degree2_tour_from_scores(score, D_single, root=root)



def _gumbel_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=shape)
    return -np.log(-np.log(u))



def _symmetric_gumbel_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros((n, n), dtype=np.float64)
    iu, ju = np.triu_indices(n, k=1)
    vals = _gumbel_noise((len(iu),), rng)
    noise[iu, ju] = vals
    noise[ju, iu] = vals
    return noise



def map_repair_seed(
    C_mod_single: np.ndarray,
    D_single: np.ndarray,
    root: int = 0,
    twoopt_passes: int = 0,
    mu_single: np.ndarray | None = None,
    repair_mode: str = "auto",
) -> tuple[list[int], bool]:
    full_adj, _, root_pair, _ = map_rooted_onetree_from_cmod(C_mod_single, root=root)
    return repair_rooted_onetree_to_tour(
        full_adj,
        C_mod_single,
        D_single,
        root=root,
        root_pair=root_pair,
        twoopt_passes=twoopt_passes,
        mu_single=mu_single,
        repair_mode=repair_mode,
    )



def _perturb_cmod(
    C_mod_single: np.ndarray,
    root: int,
    rng: np.random.Generator,
    tree_gumbel_scale: float = 0.35,
    root_gumbel_scale: float = 0.20,
    score_gumbel_scale: float = 0.0,
) -> np.ndarray:
    """Symmetric Gumbel perturbation of C_mod for best-of-K candidate generation."""
    C = np.asarray(C_mod_single, dtype=np.float64).copy()
    n = C.shape[0]
    root = int(root)
    noise = _symmetric_gumbel_noise(n, rng)
    scale = np.full((n, n), float(tree_gumbel_scale + score_gumbel_scale), dtype=np.float64)
    scale[root, :] = float(root_gumbel_scale + score_gumbel_scale)
    scale[:, root] = float(root_gumbel_scale + score_gumbel_scale)
    np.fill_diagonal(scale, 0.0)
    return C + scale * noise





def _sample_pair_from_root_logits(log_a: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Exact sample of the two root neighbors from the closed-form pair law."""
    log_a = np.asarray(log_a, dtype=np.float64)
    m = log_a.shape[0]
    if m < 2:
        raise ValueError("Need at least two non-root nodes to sample a rooted 1-tree.")
    iu, ju = np.triu_indices(m, k=1)
    logits = log_a[iu] + log_a[ju]
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    pick = int(rng.choice(len(iu), p=probs))
    return int(iu[pick]), int(ju[pick])


def _sample_weighted_neighbor(weights: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.asarray(weights, dtype=np.float64).copy()
    probs[~np.isfinite(probs)] = 0.0
    probs = np.clip(probs, a_min=0.0, a_max=None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Encountered a zero/invalid transition row in weighted Wilson sampling.")
    probs /= total
    return int(rng.choice(probs.shape[0], p=probs))


def _sample_weighted_spanning_tree_wilson(
    W: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Exact weighted spanning-tree sample via Wilson's algorithm on conductances."""
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    if n <= 1:
        return []
    if W.shape[1] != n:
        raise ValueError("W must be square.")
    if not np.all(np.isfinite(W)):
        raise RuntimeError("Weighted Wilson sampler received non-finite edge weights.")

    in_tree = np.zeros(n, dtype=bool)
    root = 0
    in_tree[root] = True
    sampled: list[tuple[int, int]] = []
    start_order = rng.permutation(n)

    for start in start_order.tolist():
        if in_tree[start]:
            continue
        path = [int(start)]
        loc = {int(start): 0}
        cur = int(start)
        max_steps = max(1000, 20 * n * n)
        steps = 0

        while not in_tree[cur]:
            row = W[cur].copy()
            row[cur] = 0.0
            nxt = _sample_weighted_neighbor(row, rng)
            if in_tree[nxt]:
                path.append(int(nxt))
                break
            if nxt in loc:
                cut = loc[nxt]
                for node in path[cut + 1 :]:
                    loc.pop(int(node), None)
                path = path[: cut + 1]
            else:
                path.append(int(nxt))
                loc[int(nxt)] = len(path) - 1
            cur = int(path[-1])
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Weighted Wilson sampler exceeded its step budget.")

        for i in range(len(path) - 1):
            u = int(path[i])
            v = int(path[i + 1])
            if not in_tree[u]:
                in_tree[u] = True
                sampled.append((u, v))

    if len(sampled) != n - 1:
        raise RuntimeError(f"Weighted Wilson sampler returned {len(sampled)} edges; expected {n - 1}.")
    return sampled


def _permute_root_to_zero(C: np.ndarray, root: int) -> tuple[np.ndarray, list[int], dict[int, int]]:
    root = int(root)
    n = C.shape[0]
    if root < 0 or root >= n:
        raise ValueError(f"root index {root} out of range for n={n}.")
    perm = [root] + [i for i in range(n) if i != root]
    inv = {new_i: old_i for new_i, old_i in enumerate(perm)}
    C_perm = C[np.ix_(perm, perm)]
    return C_perm, perm, inv


def _scaled_weight_matrix_from_cmod(
    C_mod_single: np.ndarray,
    tau: float,
    root: int = 0,
) -> tuple[np.ndarray, list[int], dict[int, int], np.ndarray, np.ndarray]:
    """Shared exact Gibbs ingredients for rooted 1-tree decoding on CPU."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 nodes for rooted 1-tree operations.")
    C_perm, perm, inv = _permute_root_to_zero(C, root=root)
    offdiag = ~np.eye(n, dtype=bool)
    shift = float(np.min(C_perm[offdiag]))
    scaled = np.clip(-(C_perm - shift) / float(max(tau, 1e-8)), -60.0, 60.0)
    W = np.exp(scaled) * offdiag.astype(np.float64)
    W = 0.5 * (W + W.T)
    return C_perm, perm, inv, scaled, W


def map_nonroot_tree_edges_from_cmod(
    C_mod_single: np.ndarray,
    root: int = 0,
) -> list[tuple[int, int]]:
    """Deterministic MAP tree on the non-root graph under C_mod."""
    C = np.asarray(C_mod_single, dtype=np.float64)
    n = C.shape[0]
    C_perm, _, inv = _permute_root_to_zero(C, root=root)
    adj_nr = prim_minimum_spanning_tree(C_perm[1:, 1:], root=0)
    rows, cols = np.where(np.triu(adj_nr, k=1) > 0)
    edges: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        edges.append((int(inv[i + 1]), int(inv[j + 1])))
    if len(edges) != n - 2:
        raise RuntimeError(f"Expected {n - 2} non-root tree edges, got {len(edges)}.")
    return edges


def sample_nonroot_tree_edges_from_cmod(
    C_mod_single: np.ndarray,
    tau: float,
    root: int = 0,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    r"""Exact weighted spanning-tree sample on V\{root}."""
    if rng is None:
        rng = np.random.default_rng()
    _, _, inv, _, W = _scaled_weight_matrix_from_cmod(C_mod_single, tau=tau, root=root)
    W_nr = W[1:, 1:]
    nr_edges_perm = _sample_weighted_spanning_tree_wilson(W_nr, rng)
    return [(int(inv[u + 1]), int(inv[v + 1])) for (u, v) in nr_edges_perm]


def assemble_rooted_onetree_from_tree_and_pair(
    n: int,
    root: int,
    tree_edges: list[tuple[int, int]],
    root_pair: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Combine a non-root tree with a chosen root pair into a rooted 1-tree."""
    a, b = int(root_pair[0]), int(root_pair[1])
    root = int(root)
    if a == b or a == root or b == root:
        raise ValueError(f"Invalid root pair {root_pair} for root {root}.")
    edges = list(tree_edges) + [(root, a), (root, b)]
    return _adjacency_from_edges(n, edges), edges


def select_root_pair_candidates(
    C_mod_single: np.ndarray,
    root: int = 0,
    pair_prob_single: np.ndarray | None = None,
    k: int = 1,
) -> list[tuple[int, int]]:
    """Deterministic distinct root-pair branches ordered by the exact pair factor."""
    k = max(int(k), 0)
    if k <= 0:
        return []
    n = int(np.asarray(C_mod_single).shape[0])
    _, _, map_pair, _ = map_rooted_onetree_from_cmod(C_mod_single, root=root)
    map_pair = tuple(sorted((int(map_pair[0]), int(map_pair[1]))))
    selected: list[tuple[int, int]] = [map_pair]
    seen = {map_pair}

    scored: list[tuple[float, float, int, int]] = []
    if pair_prob_single is not None:
        P = np.asarray(pair_prob_single, dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                if u == root or v == root:
                    continue
                prob = float(max(P[u, v], P[v, u]))
                root_cost = float(C_mod_single[root, u] + C_mod_single[root, v])
                scored.append((prob, -root_cost, int(u), int(v)))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if u == root or v == root:
                    continue
                root_cost = float(C_mod_single[root, u] + C_mod_single[root, v])
                scored.append((-root_cost, 0.0, int(u), int(v)))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    for _, _, u, v in scored:
        pair = tuple(sorted((int(u), int(v))))
        if pair in seen:
            continue
        selected.append(pair)
        seen.add(pair)
        if len(selected) >= k:
            break
    return selected[:k]


def sample_rooted_onetree_edges_from_cmod(
    C_mod_single: np.ndarray,
    tau: float,
    root: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Exact sample from the rooted 1-tree Gibbs law defined by C_mod and tau."""
    if rng is None:
        rng = np.random.default_rng()
    C_perm, _, inv, scaled, W = _scaled_weight_matrix_from_cmod(C_mod_single, tau=tau, root=root)
    i_nr, j_nr = _sample_pair_from_root_logits(scaled[0, 1:], rng)
    root_pair_perm = (i_nr + 1, j_nr + 1)
    W_nr = W[1:, 1:]
    nr_edges_perm = _sample_weighted_spanning_tree_wilson(W_nr, rng)
    edges_perm = [(0, root_pair_perm[0]), (0, root_pair_perm[1])]
    edges_perm.extend((u + 1, v + 1) for (u, v) in nr_edges_perm)

    def _map_back(edge: tuple[int, int]) -> tuple[int, int]:
        u, v = edge
        return int(inv[int(u)]), int(inv[int(v)])

    edges = [_map_back(e) for e in edges_perm]
    root_pair = _map_back((0, root_pair_perm[0]))[1], _map_back((0, root_pair_perm[1]))[1]
    return edges, (int(root_pair[0]), int(root_pair[1]))


def _adjacency_from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def deterministic_decode_candidate_tours(
    mu_single: np.ndarray,
    C_mod_single: np.ndarray,
    cand_mask_single: np.ndarray | None,
    D_single: np.ndarray,
    root: int = 0,
    twoopt_passes: int = 6,
) -> tuple[list[list[int]], list[float], dict[str, float]]:
    """Deterministic decode portfolio used by both plain decode and best-of-K.

    The two strongest deterministic seeds in this codebase are μ-greedy degree-2
    extraction and MAP rooted-1-tree repair. We improve both with the same LK-lite
    stack and choose the best resulting tour.
    """
    candidate_sets = build_move_candidates(mu_single, C_mod_single, cand_mask_single, extra_topk=5)
    tours: list[list[int]] = []
    costs: list[float] = []
    seen: set[tuple[int, ...]] = set()
    mu_fallback = 0
    map_repair_used = 0

    def _add_candidate(seed_tour: list[int]) -> None:
        improved = lk_lite_improve_tour(seed_tour, D_single, candidate_sets, twoopt_passes=twoopt_passes)
        improved = _rotate_tour_to_root(improved, root=root)
        key = tuple(int(v) for v in improved)
        if key in seen:
            return
        seen.add(key)
        tours.append(improved)
        costs.append(_tour_cost_numpy(improved, D_single))

    seed_tour, used_fallback = mu_greedy_seed(mu_single, D_single, root=root)
    mu_fallback += int(used_fallback)
    _add_candidate(seed_tour)

    map_tour, used_repair = map_repair_seed(C_mod_single, D_single, root=root, twoopt_passes=0)
    map_repair_used += int(used_repair)
    _add_candidate(map_tour)

    if not tours:
        raise RuntimeError("Deterministic decode portfolio failed to produce any tour.")
    info = {
        "fallback_rate": float(mu_fallback),
        "map_repair_rate": float(map_repair_used),
        "num_unique": float(len(tours)),
        "proposal": "best_of{mu_greedy,map_repair}+candidate_restricted_lk_lite",
    }
    return tours, costs, info


def decode_tours_from_cmod(
    mu: torch.Tensor,
    C_mod: torch.Tensor,
    cand_mask: torch.Tensor | None,
    D: torch.Tensor,
    root: int = 0,
    twoopt_passes: int = 6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Plain deterministic decode: best of μ-greedy and MAP-repair, both improved by LK-lite."""
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    B, _, _ = C_np.shape
    tours, costs = [], []
    mu_fallback_count = 0.0
    map_repair_count = 0.0
    unique_count = 0.0

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        cand_tours, cand_costs, dinfo = deterministic_decode_candidate_tours(
            mu_np[b], C_np[b], cand_single, D_np[b], root=root, twoopt_passes=twoopt_passes,
        )
        best_idx = int(np.argmin(cand_costs))
        tours.append(cand_tours[best_idx])
        costs.append(float(cand_costs[best_idx]))
        mu_fallback_count += float(dinfo["fallback_rate"])
        map_repair_count += float(dinfo["map_repair_rate"])
        unique_count += float(dinfo["num_unique"])

    info = {
        'fallback_rate': mu_fallback_count / float(max(1, B)),
        'map_repair_rate': map_repair_count / float(max(1, B)),
        'avg_unique_candidates': unique_count / float(max(1, B)),
        'proposal': 'best_of{mu_greedy,map_repair}+candidate_restricted_lk_lite',
    }
    return np.array(tours, dtype=object), np.array(costs), info



def decode_tours_from_sampled_onetrees(
    mu: torch.Tensor,
    C_mod: torch.Tensor,
    cand_mask: torch.Tensor | None,
    D: torch.Tensor,
    tau: float,
    root: int = 0,
    num_samples: int = 8,
    num_pair_samples: int | None = None,
    num_proposals_per_pair: int = 1,
    sample_bonus: float = 5.0,
    root_pair_bonus: float = 2.5,
    root_other_penalty: float = 0.0,
    seed_base: int = 0,
    tree_gumbel_scale: float = 0.35,
    root_gumbel_scale: float = 0.20,
    score_gumbel_scale: float = 0.0,
    twoopt_passes: int = 6,
    pair_prob: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Best-of-K decode with an explicit deterministic baseline and pair-aware branches.

    Portfolio per instance:
      - deterministic baseline: the same best-of{μ-greedy, MAP-repair} logic used by
        the plain decoder;
      - for each selected root pair, proposal 0 is the conditional MAP-tree branch;
      - additional proposals for that pair are exact Wilson tree samples. Because the
        rooted 1-tree family factorizes into an independent root-pair factor and a
        non-root spanning-tree factor, these are exact conditional proposals.
    """
    del sample_bonus, root_pair_bonus, root_other_penalty, tree_gumbel_scale, root_gumbel_scale, score_gumbel_scale, num_samples
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    pair_np = None if pair_prob is None else pair_prob.detach().cpu().numpy()
    B, n, _ = C_np.shape

    pair_count = max(int(num_pair_samples) if num_pair_samples is not None else 0, 0)
    proposals_per_pair = max(int(num_proposals_per_pair), 0)
    if pair_count <= 0 or proposals_per_pair <= 0:
        pair_count = 0
        proposals_per_pair = 0

    tours_best, mean_costs, best_costs = [], [], []
    repair_count = 0
    exact_count = 0
    sampler_fail_count = 0
    sampled_attempts = 0
    total_branches = 0
    unique_portfolio_size = 0.0

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        det_tours, _, _ = deterministic_decode_candidate_tours(
            mu_np[b], C_np[b], cand_single, D_np[b], root=root, twoopt_passes=twoopt_passes,
        )
        portfolio_tours: list[list[int]] = []
        portfolio_costs: list[float] = []
        seen: set[tuple[int, ...]] = set()

        def _add_portfolio_tour(tour: list[int]) -> None:
            key = tuple(int(v) for v in tour)
            if key in seen:
                return
            seen.add(key)
            portfolio_tours.append(tour)
            portfolio_costs.append(_tour_cost_numpy(tour, D_np[b]))

        for tour in det_tours:
            _add_portfolio_tour(tour)
            total_branches += 1

        selected_pairs = select_root_pair_candidates(
            C_np[b],
            root=root,
            pair_prob_single=None if pair_np is None else pair_np[b],
            k=pair_count,
        )
        map_tree_edges = map_nonroot_tree_edges_from_cmod(C_np[b], root=root)
        candidate_sets = build_move_candidates(mu_np[b], C_np[b], cand_single, extra_topk=5)

        for pair_idx, root_pair in enumerate(selected_pairs):
            for prop_idx in range(max(proposals_per_pair, 1)):
                total_branches += 1
                sampled_attempts += 1
                if prop_idx == 0:
                    full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, map_tree_edges, root_pair)
                else:
                    rng = np.random.default_rng(int(seed_base + 1000003 * b + 9176 * (pair_idx + 1) + 131 * prop_idx + 53))
                    try:
                        sampled_tree_edges = sample_nonroot_tree_edges_from_cmod(C_np[b], tau=float(tau), root=root, rng=rng)
                        full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, sampled_tree_edges, root_pair)
                    except RuntimeError:
                        sampler_fail_count += 1
                        full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(n, root, map_tree_edges, root_pair)

                tour, used_repair = repair_rooted_onetree_to_tour(
                    full_adj,
                    C_np[b],
                    D_np[b],
                    root=root,
                    root_pair=root_pair,
                    twoopt_passes=0,
                )
                repair_count += int(used_repair)
                exact_count += int(not used_repair)
                improved = lk_lite_improve_tour(tour, D_np[b], candidate_sets, twoopt_passes=twoopt_passes)
                improved = _rotate_tour_to_root(improved, root=root)
                _add_portfolio_tour(improved)

        if not portfolio_tours:
            raise RuntimeError("Sampled decoder portfolio is empty.")
        unique_portfolio_size += float(len(portfolio_tours))
        mean_costs.append(float(np.mean(portfolio_costs)))
        best_idx = int(np.argmin(portfolio_costs))
        best_costs.append(float(portfolio_costs[best_idx]))
        tours_best.append(portfolio_tours[best_idx])

    info = {
        'fallback_rate': float(repair_count) / float(max(1, sampled_attempts)),
        'exact_rate': float(exact_count) / float(max(1, sampled_attempts)),
        'sampler_fail_rate': float(sampler_fail_count) / float(max(1, sampled_attempts)),
        'avg_unique_portfolio_size': unique_portfolio_size / float(max(1, B)),
        'proposal': 'deterministic_baseline + top_pair_conditional_map/tree_samples + candidate_restricted_lk_lite',
        'num_pair_samples': float(pair_count),
        'num_proposals_per_pair': float(proposals_per_pair),
        'total_branches': float(total_branches) / float(max(1, B)),
    }
    return np.array(tours_best, dtype=object), np.array(mean_costs), np.array(best_costs), info


# ================================================================
# §7b  Ablation decoder: run 4 strategies independently, report each
# ================================================================


def _tour_key(tour: list[int]) -> tuple[int, ...]:
    """Hashable key for a tour, rotation-invariant via min-index normalization."""
    if not tour:
        return tuple()
    n = len(tour)
    k = int(np.argmin(np.asarray(tour)))
    rot = tour[k:] + tour[:k]
    # Also canonicalize direction.
    if n >= 3 and rot[1] > rot[-1]:
        rot = [rot[0]] + rot[:0:-1]
    return tuple(int(v) for v in rot)


def _eval_seed(
    seed_tour: list[int],
    D_single: np.ndarray,
    candidate_sets: list[set[int]],
    root: int,
    twoopt_passes: int,
    C_mod_single: np.ndarray | None = None,
) -> tuple[list[int], float, list[int], float]:
    """Return (raw_tour, raw_cost, lk_tour, lk_cost) for one seed.

    If C_mod_single is provided, LK-alpha ordering is used inside lk_lite_improve_tour.
    """
    raw_tour = _rotate_tour_to_root(list(seed_tour), root=root)
    raw_cost = float(_tour_cost_numpy(raw_tour, D_single))
    if twoopt_passes > 0:
        lk_tour = lk_lite_improve_tour(raw_tour, D_single, candidate_sets, twoopt_passes=twoopt_passes, C_mod=C_mod_single)
        lk_tour = _rotate_tour_to_root(lk_tour, root=root)
        lk_cost = float(_tour_cost_numpy(lk_tour, D_single))
    else:
        lk_tour = raw_tour
        lk_cost = raw_cost
    return raw_tour, raw_cost, lk_tour, lk_cost


def _greedy_degree2_with_forced_pair(
    score: np.ndarray,
    D: np.ndarray,
    root: int,
    forced_pair: tuple[int, int],
) -> tuple[list[int], bool]:
    """Like greedy_degree2_tour_from_scores but forces the root pair edges.

    Forces (root, a) and (root, b) into the matching first, then greedily
    completes the degree-2 Hamiltonian cycle using `score` (with D as tiebreak).
    """
    n = score.shape[0]
    a, b = int(forced_pair[0]), int(forced_pair[1])
    root = int(root)
    if a == b or a == root or b == root:
        raise ValueError(f"Invalid forced pair {forced_pair} for root {root}.")

    deg = np.zeros(n, dtype=int)
    dsu = _DSU(n)
    selected: list[tuple[int, int]] = []

    # Force the two root-incident edges.
    for (u, v) in ((root, a), (root, b)):
        if dsu.find(u) == dsu.find(v):
            # Shouldn't happen with fresh DSU, but guard anyway.
            break
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    all_edges: list[tuple[float, float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if i == root or j == root:
                # Root's two edges are already fixed.
                continue
            all_edges.append((float(score[i, j]), -float(D[i, j]), i, j))
    all_edges.sort(reverse=True)

    for _, _, u, v in all_edges:
        if deg[u] >= 2 or deg[v] >= 2:
            continue
        same = dsu.find(u) == dsu.find(v)
        if same:
            if len(selected) == n - 1 and deg[u] == 1 and deg[v] == 1:
                selected.append((u, v))
                deg[u] += 1
                deg[v] += 1
                break
            continue
        dsu.union(u, v)
        selected.append((u, v))
        deg[u] += 1
        deg[v] += 1

    if len(selected) == n:
        tour = cycle_from_degree2_edges(selected, n=n, root=root)
        if tour is not None:
            return tour, False

    # Fallback: drop the forcing constraint and run the unconstrained greedy.
    return greedy_degree2_tour_from_scores(score, D, root=root)


def decode_tours_ablation(
    mu: torch.Tensor,
    C_mod: torch.Tensor,
    cand_mask: torch.Tensor | None,
    D: torch.Tensor,
    root: int = 0,
    twoopt_passes: int = 6,
    num_root_pairs: int = 4,
    num_gumbel_draws: int = 8,
    gumbel_scale: float = 0.20,
    seed_base: int = 0,
    pair_prob: torch.Tensor | None = None,
    use_lk_alpha: bool = False,
) -> Dict[str, np.ndarray | Dict]:
    r"""Run four decode strategies independently and report each plus the envelope.

    Strategies (each yields one or more tour candidates, all optionally LK-lite improved):
      S1 mu_greedy       : deterministic degree-2 greedy from \log \mu^{sym} with D tiebreak.
      S2 map_repair      : deterministic MAP rooted 1-tree + backbone-insertion repair.
      S3 rootpair_mapK   : top-K root pairs ranked by pair_prob (or root_cost fallback);
                            for each pair, use MAP non-root tree and repair.
      S4 gumbel_muM      : M independent Gumbel-perturbed mu-greedy degree-2 tours.

    Portfolio: best cost across all tours from S1..S4.
    Attribution: which strategy provided the portfolio winner per instance.

    Returns a dict keyed by strategy name with np arrays of length B for raw_cost_best
    and lk_cost_best (plus best tours), and 'portfolio' / 'attribution' / 'info'.
    """
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    pair_np = None if pair_prob is None else pair_prob.detach().cpu().numpy()
    B, n, _ = C_np.shape

    K = max(0, int(num_root_pairs))
    M = max(0, int(num_gumbel_draws))

    # Per-strategy arrays.
    names = ["s1_mu_greedy", "s2_map_repair", "s3_rootpair_mapK", "s4_gumbel_muM"]
    results: Dict[str, Dict[str, list]] = {
        name: {"raw_tour": [], "raw_cost": [], "lk_tour": [], "lk_cost": []} for name in names
    }
    portfolio_tours: list[list[int]] = []
    portfolio_costs: list[float] = []
    attribution: list[str] = []

    # Per-strategy aggregate info.
    info_counts = {
        "s1_fallback_frac": 0.0,
        "s2_repair_frac": 0.0,
        "s3_repair_frac": 0.0,
        "s3_num_pairs_mean": 0.0,
        "s4_unique_frac": 0.0,  # mean unique tours / M
        "s4_draws": float(M),
    }

    for b in range(B):
        cand_single = None if cand_np is None else cand_np[b]
        mu_b = mu_np[b]
        C_b = C_np[b]
        D_b = D_np[b]
        pair_b = None if pair_np is None else pair_np[b]

        # Shared candidate sets for LK-lite across all strategies for this instance.
        candidate_sets = build_move_candidates(mu_b, C_b, cand_single, extra_topk=5)
        # LK-alpha: if enabled, pass C_mod to the inner moves so candidate ordering
        # follows the learned Lagrangian gradient. If disabled, C_mod_for_lk is None
        # and LK behavior is bit-for-bit identical to v6.
        C_mod_for_lk = C_b if use_lk_alpha else None

        best_cost_b = float("inf")
        best_tour_b: list[int] = []
        best_strategy_b = ""

        # ---------- S1: mu-greedy ----------
        seed1, used_fb1 = mu_greedy_seed(mu_b, D_b, root=root)
        info_counts["s1_fallback_frac"] += float(used_fb1)
        raw_t, raw_c, lk_t, lk_c = _eval_seed(seed1, D_b, candidate_sets, root, twoopt_passes, C_mod_single=C_mod_for_lk)
        results["s1_mu_greedy"]["raw_tour"].append(raw_t)
        results["s1_mu_greedy"]["raw_cost"].append(raw_c)
        results["s1_mu_greedy"]["lk_tour"].append(lk_t)
        results["s1_mu_greedy"]["lk_cost"].append(lk_c)
        if lk_c < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = lk_c, lk_t, "s1_mu_greedy"

        # ---------- S2: MAP rooted 1-tree + repair ----------
        map_full_adj, _, map_root_pair, _ = map_rooted_onetree_from_cmod(C_b, root=root)
        seed2, used_repair2 = repair_rooted_onetree_to_tour(
            map_full_adj, C_b, D_b, root=root, root_pair=map_root_pair, twoopt_passes=0,
        )
        info_counts["s2_repair_frac"] += float(used_repair2)
        raw_t, raw_c, lk_t, lk_c = _eval_seed(seed2, D_b, candidate_sets, root, twoopt_passes, C_mod_single=C_mod_for_lk)
        results["s2_map_repair"]["raw_tour"].append(raw_t)
        results["s2_map_repair"]["raw_cost"].append(raw_c)
        results["s2_map_repair"]["lk_tour"].append(lk_t)
        results["s2_map_repair"]["lk_cost"].append(lk_c)
        if lk_c < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = lk_c, lk_t, "s2_map_repair"

        # ---------- S3: top-K root-pair conditional MAP trees ----------
        s3_best_raw, s3_best_lk = float("inf"), float("inf")
        s3_best_raw_tour: list[int] = []
        s3_best_lk_tour: list[int] = []
        if K > 0:
            selected_pairs = select_root_pair_candidates(
                C_b, root=root, pair_prob_single=pair_b, k=K,
            )
            # Compute the MAP non-root tree ONCE; it's independent of the root pair.
            map_tree_edges = map_nonroot_tree_edges_from_cmod(C_b, root=root)
            s3_repair_count = 0
            s3_branches = 0
            for root_pair in selected_pairs:
                s3_branches += 1
                full_adj, _ = assemble_rooted_onetree_from_tree_and_pair(
                    n, root, map_tree_edges, root_pair,
                )
                seed3, used_repair3 = repair_rooted_onetree_to_tour(
                    full_adj, C_b, D_b, root=root, root_pair=root_pair, twoopt_passes=0,
                )
                s3_repair_count += int(used_repair3)
                raw_t, raw_c, lk_t, lk_c = _eval_seed(seed3, D_b, candidate_sets, root, twoopt_passes, C_mod_single=C_mod_for_lk)
                if raw_c < s3_best_raw:
                    s3_best_raw, s3_best_raw_tour = raw_c, raw_t
                if lk_c < s3_best_lk:
                    s3_best_lk, s3_best_lk_tour = lk_c, lk_t
            info_counts["s3_repair_frac"] += float(s3_repair_count) / float(max(1, s3_branches))
            info_counts["s3_num_pairs_mean"] += float(s3_branches)
        else:
            # If K=0, report NaN and skip.
            s3_best_raw = float("nan")
            s3_best_lk = float("nan")
        results["s3_rootpair_mapK"]["raw_tour"].append(s3_best_raw_tour)
        results["s3_rootpair_mapK"]["raw_cost"].append(s3_best_raw)
        results["s3_rootpair_mapK"]["lk_tour"].append(s3_best_lk_tour)
        results["s3_rootpair_mapK"]["lk_cost"].append(s3_best_lk)
        if math.isfinite(s3_best_lk) and s3_best_lk < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = s3_best_lk, s3_best_lk_tour, "s3_rootpair_mapK"

        # ---------- S4: Gumbel mu-greedy (M draws) ----------
        s4_best_raw, s4_best_lk = float("inf"), float("inf")
        s4_best_raw_tour: list[int] = []
        s4_best_lk_tour: list[int] = []
        if M > 0:
            rng = np.random.default_rng(int(seed_base + 1000003 * b + 31))
            base_score = _mu_score_matrix(mu_b)
            seen_keys: set[tuple[int, ...]] = set()
            unique = 0
            for m in range(M):
                noise = _symmetric_gumbel_noise(n, rng)
                score_m = base_score + float(gumbel_scale) * noise
                np.fill_diagonal(score_m, -np.inf)
                seed4, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
                raw_t, raw_c, lk_t, lk_c = _eval_seed(seed4, D_b, candidate_sets, root, twoopt_passes, C_mod_single=C_mod_for_lk)
                k_raw = _tour_key(raw_t)
                if k_raw not in seen_keys:
                    seen_keys.add(k_raw)
                    unique += 1
                if raw_c < s4_best_raw:
                    s4_best_raw, s4_best_raw_tour = raw_c, raw_t
                if lk_c < s4_best_lk:
                    s4_best_lk, s4_best_lk_tour = lk_c, lk_t
            info_counts["s4_unique_frac"] += float(unique) / float(max(1, M))
        else:
            s4_best_raw = float("nan")
            s4_best_lk = float("nan")
        results["s4_gumbel_muM"]["raw_tour"].append(s4_best_raw_tour)
        results["s4_gumbel_muM"]["raw_cost"].append(s4_best_raw)
        results["s4_gumbel_muM"]["lk_tour"].append(s4_best_lk_tour)
        results["s4_gumbel_muM"]["lk_cost"].append(s4_best_lk)
        if math.isfinite(s4_best_lk) and s4_best_lk < best_cost_b:
            best_cost_b, best_tour_b, best_strategy_b = s4_best_lk, s4_best_lk_tour, "s4_gumbel_muM"

        portfolio_costs.append(best_cost_b)
        portfolio_tours.append(best_tour_b)
        attribution.append(best_strategy_b)

    # Finalize.
    Bf = float(max(1, B))
    info = {
        "s1_fallback_frac": info_counts["s1_fallback_frac"] / Bf,
        "s2_repair_frac": info_counts["s2_repair_frac"] / Bf,
        "s3_repair_frac": info_counts["s3_repair_frac"] / Bf if K > 0 else 0.0,
        "s3_num_pairs_mean": info_counts["s3_num_pairs_mean"] / Bf if K > 0 else 0.0,
        "s4_unique_frac": info_counts["s4_unique_frac"] / Bf if M > 0 else 0.0,
        "s4_draws": float(M),
        "num_root_pairs": float(K),
        "num_gumbel_draws": float(M),
        "gumbel_scale": float(gumbel_scale),
        "twoopt_passes": float(twoopt_passes),
        "lk_alpha": float(bool(use_lk_alpha)),
    }

    out: Dict[str, np.ndarray | Dict] = {}
    for name in names:
        out[name] = {
            "raw_tour": np.array(results[name]["raw_tour"], dtype=object),
            "raw_cost": np.array(results[name]["raw_cost"], dtype=np.float64),
            "lk_tour": np.array(results[name]["lk_tour"], dtype=object),
            "lk_cost": np.array(results[name]["lk_cost"], dtype=np.float64),
        }
    out["portfolio"] = {
        "tour": np.array(portfolio_tours, dtype=object),
        "cost": np.array(portfolio_costs, dtype=np.float64),
    }
    out["attribution"] = np.array(attribution, dtype=object)
    out["info"] = info
    return out



def _generate_decode_noise(
    noise_type: str,
    n: int,
    root: int,
    rng: np.random.Generator,
    mu_b: np.ndarray | None = None,
    C_mod_b: np.ndarray | None = None,
    tau: float = 0.2,
) -> np.ndarray:
    """Generate symmetric (n, n) perturbation noise for one decode draw.

    Types:
      gumbel                 : symmetric Gumbel(0,1) — heavy right tail, original v9 behavior
      gaussian               : symmetric N(0,1) — no tail bias
      uncertainty            : N(0,1) scaled by mu*(1-mu) per edge — explores uncertain edges only
      dual                   : N(0,1) at node level, broadcast to edges as delta_i + delta_j
      covariance             : |phi_i - phi_j| with phi ~ N(0, L^{-1}) — Kirchhoff magnitude,
                               large on high effective resistance edges (topologically ambiguous).
                               Normalized to std 1 overall.
      covariance_uncertainty : covariance noise further weighted by sqrt(mu(1-mu)) per edge —
                               combines graph-topology scale (covariance) with model-uncertainty
                               targeting (only perturbs edges model is still unsure about).
    """
    if noise_type == "gumbel":
        return _symmetric_gumbel_noise(n, rng)
    elif noise_type == "gaussian":
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise
    elif noise_type == "uncertainty":
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        if mu_b is not None:
            mu_sym = 0.5 * (mu_b + mu_b.T)
            unc = np.clip(mu_sym, 0.0, 1.0) * (1.0 - np.clip(mu_sym, 0.0, 1.0))
            # Normalize so max uncertainty edge gets scale 1.0
            unc_max = unc.max()
            if unc_max > 1e-12:
                unc = unc / unc_max
            noise = noise * unc
        return noise
    elif noise_type == "dual":
        delta = rng.standard_normal(n)
        delta[root] = 0.0
        noise = delta[:, None] + delta[None, :]
        np.fill_diagonal(noise, 0.0)
        return noise
    elif noise_type == "covariance":
        return _covariance_noise(n, root, rng, C_mod_b=C_mod_b, tau=tau,
                                 mu_b=mu_b, uncertainty_weight=False)
    elif noise_type == "covariance_uncertainty":
        return _covariance_noise(n, root, rng, C_mod_b=C_mod_b, tau=tau,
                                 mu_b=mu_b, uncertainty_weight=True)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}. "
                         f"Use gumbel/gaussian/uncertainty/dual/covariance/covariance_uncertainty.")


def _covariance_noise(
    n: int,
    root: int,
    rng: np.random.Generator,
    C_mod_b: np.ndarray | None = None,
    tau: float = 0.2,
    mu_b: np.ndarray | None = None,
    uncertainty_weight: bool = False,
) -> np.ndarray:
    """Sample structurally coherent noise from the 1-tree Gibbs covariance.

    The weighted Laplacian L on V\\{root} defines a Gaussian field
        phi ~ N(0, L^{-1}), Cov(phi_i, phi_j) = L^{-1}_{ij}.
    Edge perturbations are the Kirchhoff form noise_ij = phi_i - phi_j, with
        Var(noise_ij) = R^eff_ij = L^{-1}_{ii} + L^{-1}_{jj} - 2 L^{-1}_{ij}
        Cov(noise_ij, noise_ik) = L^{-1}_{ii} - L^{-1}_{ij} - L^{-1}_{ik} + L^{-1}_{jk}
    i.e. competing edges at a shared vertex get anti-correlated noise, and
    magnitude scales with effective resistance.

    The decoder expects symmetric noise matrices, so we emit |phi_i - phi_j|.
    This preserves the resistance-scaled magnitude structure (large on
    topologically ambiguous edges) but loses the anti-correlation sign. For
    best-of-K sampling via score perturbation, magnitude-based exploration
    is what matters — the sign is washed out by the max anyway.

    Fixes in v17+ vs v16:
      - noise = |phi_i - phi_j| (Kirchhoff magnitude), NOT phi_i + phi_j
        (which was node-additive and absorbed by lambda, carrying no edge
        signal).
      - No std-normalization on phi: the heterogeneous per-node variances
        ARE the structural signal. Normalizing phi to unit std destroys it.
      - L is rescaled by its mean diagonal to keep Cholesky well-conditioned
        across cost scales, without flattening relative variances.
      - Final noise matrix is normalized to unit std overall (not per-node),
        so gumbel_scale retains interpretable magnitude across noise types.

    Optional: set uncertainty_weight=True to multiply by sqrt(mu(1-mu)), 
    concentrating structural noise on edges the model is still uncertain 
    about. Combines graph topology with marginal uncertainty.

    Cost: one Cholesky O((n-1)^3) + one triangular solve O((n-1)^2).
    """
    if C_mod_b is None:
        # Fallback to gaussian if C_mod not available
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise

    tau_safe = max(float(tau), 1e-6)
    m = n - 1

    nonroot = [i for i in range(n) if i != root]
    nonroot_arr = np.array(nonroot)

    # Build weighted Laplacian on V\{root}, grounded by root-edge weights on
    # diagonal (standard non-root rooted-spanning-tree Laplacian).
    C_sym = 0.5 * (C_mod_b + C_mod_b.T)
    C_sub = C_sym[np.ix_(nonroot_arr, nonroot_arr)]
    W_sub = np.exp(-C_sub / tau_safe)
    W_sub = np.clip(W_sub, 1e-30, 1e30)
    np.fill_diagonal(W_sub, 0.0)
    L = np.diag(W_sub.sum(axis=1)) - W_sub
    root_w = np.exp(-C_sym[nonroot_arr, root] / tau_safe)
    root_w = np.clip(root_w, 1e-30, 1e30)
    L = L + np.diag(root_w)

    # Rescale L by its mean diagonal: preserves relative per-node variance,
    # keeps Cholesky numerically stable across cost magnitudes.
    L_diag_mean = max(float(np.mean(np.diag(L))), 1e-8)
    L_scaled = L / L_diag_mean

    # Cholesky: L_scaled = R^T R
    try:
        R = np.linalg.cholesky(L_scaled + 1e-6 * np.eye(m))
    except np.linalg.LinAlgError:
        raw = rng.standard_normal((n, n))
        noise = 0.5 * (raw + raw.T)
        np.fill_diagonal(noise, 0.0)
        return noise

    # Sample phi ~ N(0, L_scaled^{-1}):  phi = R^{-T} z,  z ~ N(0, I)
    z = rng.standard_normal(m)
    phi_local = np.linalg.solve(R.T, z)

    # Embed back to full n nodes (root gets phi=0 by grounding)
    phi = np.zeros(n, dtype=np.float64)
    phi[nonroot_arr] = phi_local

    # Symmetric Kirchhoff-magnitude edge noise:
    noise = np.abs(phi[:, None] - phi[None, :])
    np.fill_diagonal(noise, 0.0)

    # Optional uncertainty weighting: concentrate on edges model is uncertain about
    if uncertainty_weight and mu_b is not None:
        mu_sym = 0.5 * (mu_b + mu_b.T)
        mu_c = np.clip(mu_sym, 0.0, 1.0)
        unc = mu_c * (1.0 - mu_c)
        unc_max = float(unc.max()) if unc.size else 0.0
        if unc_max > 1e-10:
            noise = noise * np.sqrt(unc / unc_max)
        np.fill_diagonal(noise, 0.0)

    # Normalize so overall noise std is 1 — keeps gumbel_scale comparable
    # across noise types. Relative structure (what carries signal) is preserved.
    offdiag = noise[~np.eye(n, dtype=bool)]
    ns_std = float(offdiag.std())
    if ns_std > 1e-10:
        noise = noise / ns_std
    np.fill_diagonal(noise, 0.0)
    return noise


# =========================================================================
# v17+: Hybrid CPU/GPU decoder — batched noise + batched Prim's on GPU,
# per-instance greedy/repair/cycle-extraction on CPU.
# =========================================================================

def _batched_prim_gpu(cost: torch.Tensor, root: int = 0) -> torch.Tensor:
    """Batched Prim's MST on GPU.

    Args:
      cost: (B, n, n) symmetric cost matrix on GPU.
      root: starting vertex for Prim's.

    Returns:
      adj: (B, n, n) int64 adjacency in {0,1} on GPU. Each row corresponds to
           the MST of one instance. Diagonal is zero.

    Complexity: O(n) GPU kernel launches, each doing O(B·n) elementwise work.
    Replaces B serial CPU Prim's calls with n batched GPU ops.
    """
    B, n, _ = cost.shape
    device = cost.device
    INF = float("inf")
    in_tree = torch.zeros(B, n, dtype=torch.bool, device=device)
    parent = torch.full((B, n), -1, dtype=torch.long, device=device)
    best = torch.full((B, n), INF, device=device, dtype=cost.dtype)
    best[:, root] = 0.0

    batch_idx = torch.arange(B, device=device)

    for _ in range(n):
        masked = torch.where(in_tree, torch.full_like(best, INF), best)
        u = masked.argmin(dim=1)                       # (B,)
        in_tree[batch_idx, u] = True
        c_u = cost[batch_idx, u]                       # (B, n) — row u for each instance
        # Mask already-in-tree vertices out of the update
        better = (~in_tree) & (c_u < best)
        parent = torch.where(better, u.unsqueeze(1).expand_as(parent), parent)
        best = torch.where(better, c_u, best)

    # Build adjacency from parent pointers
    adj = torch.zeros(B, n, n, dtype=torch.long, device=device)
    v_idx = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)  # (B, n)
    valid = parent >= 0                                                  # (B, n) bool
    # Flatten valid entries for scatter
    b_flat = batch_idx.unsqueeze(1).expand(-1, n)[valid]                 # (K,)
    v_flat = v_idx[valid]                                                # (K,)
    p_flat = parent[valid]                                               # (K,)
    adj[b_flat, p_flat, v_flat] = 1
    adj[b_flat, v_flat, p_flat] = 1
    return adj


def _batched_noise_gpu(
    noise_type: str,
    M: int,
    B: int,
    n: int,
    root: int,
    mu_b: torch.Tensor,          # (B, n, n) marginals
    C_mod_b: torch.Tensor,       # (B, n, n) reduced cost
    tau: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate all (M, B) noise matrices on GPU.

    Returns: (M, B, n, n) noise tensor. Draw 0 is the zero-noise draw; draws
    1..M-1 are sampled.

    Noise type semantics match _generate_decode_noise for apples-to-apples
    comparability. Per-instance normalization (std=1) is applied on GPU.
    """
    # We use a separate generator to ensure reproducibility without perturbing
    # global torch rng state.
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    out = torch.zeros(M, B, n, n, device=device, dtype=dtype)
    eye_off = ~torch.eye(n, dtype=torch.bool, device=device)  # off-diagonal mask

    if noise_type == "gumbel":
        # Symmetric Gumbel(0,1) via -log(-log(U)), symmetrize
        # Only generate M-1 draws (draw 0 is zero)
        u_raw = torch.rand(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        u_raw = u_raw.clamp(min=1e-12, max=1 - 1e-12)
        raw = -torch.log(-torch.log(u_raw))
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        out[1:] = noise
    elif noise_type == "gaussian":
        raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        out[1:] = noise
    elif noise_type == "uncertainty":
        raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
        noise = 0.5 * (raw + raw.transpose(-1, -2))
        mu_sym = 0.5 * (mu_b + mu_b.transpose(-1, -2))   # (B, n, n)
        mu_c = mu_sym.clamp(0.0, 1.0)
        unc = mu_c * (1.0 - mu_c)                        # (B, n, n)
        unc_max = unc.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
        unc_scale = unc / unc_max                        # (B, n, n) in [0,1]
        noise = noise * unc_scale.unsqueeze(0)           # broadcast over M
        out[1:] = noise
    elif noise_type == "dual":
        delta = torch.randn(M - 1, B, n, device=device, generator=g, dtype=dtype)
        delta[..., root] = 0.0
        noise = delta.unsqueeze(-1) + delta.unsqueeze(-2)
        out[1:] = noise
    elif noise_type in ("covariance", "covariance_uncertainty"):
        # Batched Cholesky over B Laplacians.
        tau_safe = max(float(tau), 1e-6)
        # Build non-root index
        nonroot = [i for i in range(n) if i != root]
        nr_idx = torch.tensor(nonroot, device=device, dtype=torch.long)
        m = n - 1
        # C_mod_b symmetrized and indexed on non-root
        C_sym = 0.5 * (C_mod_b + C_mod_b.transpose(-1, -2))    # (B, n, n)
        # Gather non-root submatrix
        C_sub = C_sym.index_select(-2, nr_idx).index_select(-1, nr_idx)  # (B, m, m)
        # Weights (clip exponent to avoid overflow)
        exponent = (-C_sub / tau_safe).clamp(-60.0, 60.0)
        W_sub = torch.exp(exponent)                              # (B, m, m)
        # Zero diagonal
        W_sub = W_sub * (1.0 - torch.eye(m, device=device, dtype=dtype))
        # Laplacian
        L = torch.diag_embed(W_sub.sum(dim=-1)) - W_sub           # (B, m, m)
        # Add root-edge diagonal
        root_exponent = (-C_sym[:, nr_idx, root] / tau_safe).clamp(-60.0, 60.0)
        root_w = torch.exp(root_exponent)                        # (B, m)
        L = L + torch.diag_embed(root_w)
        # Rescale by mean diagonal
        L_diag_mean = L.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        L_diag_mean = L_diag_mean.clamp(min=1e-8)
        L_scaled = L / L_diag_mean
        # Cholesky with ridge
        ridge = 1e-6 * torch.eye(m, device=device, dtype=dtype)
        try:
            R = torch.linalg.cholesky(L_scaled + ridge)           # (B, m, m)
        except Exception:
            # Fallback to Gaussian noise on failure
            fallback_raw = torch.randn(M - 1, B, n, n, device=device, generator=g, dtype=dtype)
            fallback = 0.5 * (fallback_raw + fallback_raw.transpose(-1, -2))
            out[1:] = fallback
            return out
        # Sample phi: R^T phi = z,  z ~ N(0, I). Batched triangular solve.
        z = torch.randn(M - 1, B, m, device=device, generator=g, dtype=dtype)  # (M-1, B, m)
        # Need to solve R.transpose @ phi = z per batch, broadcast over M-1 draws
        # torch.linalg.solve_triangular: (B, m, m) @ (B, m, K) -> (B, m, K)
        # Flatten M-1 draws into the rhs dim
        z_flat = z.permute(1, 2, 0).contiguous()  # (B, m, M-1)
        phi_flat = torch.linalg.solve_triangular(
            R.transpose(-1, -2), z_flat, upper=True, unitriangular=False
        )                                          # (B, m, M-1)
        phi = phi_flat.permute(2, 0, 1)            # (M-1, B, m)
        # Embed back to full n
        phi_full = torch.zeros(M - 1, B, n, device=device, dtype=dtype)
        phi_full[:, :, nr_idx] = phi               # root stays 0
        # Kirchhoff magnitude: |phi_i - phi_j|
        noise = torch.abs(phi_full.unsqueeze(-1) - phi_full.unsqueeze(-2))  # (M-1, B, n, n)
        # Optional uncertainty weighting
        if noise_type == "covariance_uncertainty":
            mu_sym = 0.5 * (mu_b + mu_b.transpose(-1, -2))
            mu_c = mu_sym.clamp(0.0, 1.0)
            unc = mu_c * (1.0 - mu_c)
            unc_max = unc.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
            unc_scale = torch.sqrt(unc / unc_max)  # (B, n, n)
            noise = noise * unc_scale.unsqueeze(0)
        out[1:] = noise
    else:
        raise ValueError(f"Unknown noise_type for GPU path: {noise_type!r}")

    # Zero diagonals
    out = out * eye_off.unsqueeze(0).unsqueeze(0)
    # Per-(M,B) std-normalize: compute std over off-diagonal entries
    # shape (M, B, 1, 1)
    flat = out.reshape(M, B, -1)
    # off-diag count per instance
    od_count = int(eye_off.sum().item())
    # variance of off-diag entries per (m, b)
    # Use the mask: sum of squares over off-diag / od_count
    sq = (out * out).sum(dim=(-1, -2)) / max(od_count, 1)  # (M, B)
    sd = sq.clamp(min=1e-20).sqrt()                         # (M, B)
    # Avoid dividing draw 0 by (effectively 0) — guard
    sd = sd.where(sd > 1e-10, torch.ones_like(sd))
    out = out / sd.unsqueeze(-1).unsqueeze(-1)
    # Re-zero diagonal (the division may have left tiny noise on diagonal)
    out = out * eye_off.unsqueeze(0).unsqueeze(0)
    # Draw 0 stays zero — re-zero it to be safe
    out[0] = 0.0
    return out


# ---------------------------------------------------------------------------
# v19 decode parallelization: per-instance CPU work extracted to a top-level
# function so it can be sent to a ProcessPoolExecutor.
# ---------------------------------------------------------------------------
def _decode_single_instance_cpu(
    bi: int,
    M: int,
    n: int,
    root: int,
    twoopt_passes: int,
    seed_split: float,
    use_lk_alpha: bool,
    use_mu_repair: bool,
    repair_mode: str,
    mu_b: np.ndarray,
    D_b: np.ndarray,
    scores_bi: np.ndarray,       # (M, n, n)
    C_mod_bi: np.ndarray,        # (M, n, n)
    full_adj_bi: np.ndarray,     # (M, n, n)
    root_pair_bi: np.ndarray,    # (M, 2)
    cand_bi: np.ndarray | None,  # (n, n) or None
) -> dict:
    """Process a single batch instance's M Gumbel draws; pickle-safe.

    Mirrors the body of the ``for bi in range(B)`` loop in
    :func:`decode_gumbel_hybrid`.  All captured closures (``_try_seed``) have
    been flattened out so the function stands alone.
    """
    # Build candidate sets for LK
    if cand_bi is not None:
        candidate_sets: list[set[int]] = [
            {int(j) for j in range(n) if j != i and cand_bi[i, j]}
            for i in range(n)
        ]
    else:
        full = set(range(n))
        candidate_sets = [full - {i} for i in range(n)]
    C_mod_for_lk = C_mod_bi[0] if use_lk_alpha else None

    all_costs: list[float] = []
    a_costs: list[float] = []
    b_costs: list[float] = []
    best_cost_b = float("inf")
    best_tour_b: list[int] = []
    best_a_cost = float("inf")
    best_b_cost = float("inf")
    det_cost_b = float("inf")
    det_tour_b: list[int] = []
    seen_keys: set[tuple[int, ...]] = set()

    def try_seed_local(seed_tour: list[int]) -> tuple[float, list[int]]:
        nonlocal best_cost_b, best_tour_b
        _, _, lk_tour, lk_cost = _eval_seed(
            seed_tour, D_b, candidate_sets, root, twoopt_passes,
            C_mod_single=C_mod_for_lk,
        )
        all_costs.append(lk_cost)
        seen_keys.add(_tour_key(lk_tour))
        if lk_cost < best_cost_b:
            best_cost_b = lk_cost
            best_tour_b = lk_tour
        return lk_cost, lk_tour

    # Seed-split routing
    run_a_for_draw = [False] * M
    run_b_for_draw = [False] * M
    if seed_split < 0.0:
        for m_i in range(M):
            run_a_for_draw[m_i] = True
            run_b_for_draw[m_i] = True
    else:
        s = max(0.0, min(1.0, float(seed_split)))
        n_a = int(round(s * M))
        if n_a > 0 and n_a < M:
            run_a_for_draw[0] = True
            run_b_for_draw[0] = True
            remaining_a = n_a - 1
            remaining_b = M - 1 - remaining_a
            for m_i in range(1, M):
                if remaining_a > 0 and (remaining_b == 0 or
                                        ((m_i - 1) * n_a) % M < ((m_i) * n_a) % M):
                    run_a_for_draw[m_i] = True
                    remaining_a -= 1
                else:
                    run_b_for_draw[m_i] = True
                    remaining_b -= 1
        elif n_a == 0:
            for m_i in range(M):
                run_b_for_draw[m_i] = True
        else:
            for m_i in range(M):
                run_a_for_draw[m_i] = True

    for m_i in range(M):
        ca = float("inf")
        cb = float("inf")

        if run_a_for_draw[m_i]:
            score_m = scores_bi[m_i].copy()
            np.fill_diagonal(score_m, -np.inf)
            seed_a, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
            ca, tour_a = try_seed_local(seed_a)
            a_costs.append(ca)
            if ca < best_a_cost:
                best_a_cost = ca

        if run_b_for_draw[m_i]:
            full_adj_b = full_adj_bi[m_i]
            root_pair = (int(root_pair_bi[m_i, 0]), int(root_pair_bi[m_i, 1]))
            C_mod_m = C_mod_bi[m_i]
            seed_b, _ = repair_rooted_onetree_to_tour(
                full_adj_b, C_mod_m, D_b,
                root=root, root_pair=root_pair, twoopt_passes=0,
                mu_single=(mu_b if use_mu_repair else None),
                repair_mode=repair_mode,
            )
            cb, tour_b = try_seed_local(seed_b)
            b_costs.append(cb)
            if cb < best_b_cost:
                best_b_cost = cb

        if m_i == 0:
            det_cost_b = min(ca, cb)
            if ca <= cb and run_a_for_draw[m_i]:
                det_tour_b = tour_a
            elif run_b_for_draw[m_i]:
                det_tour_b = tour_b

    total_candidates = max(1, len(all_costs))
    return {
        "bi": bi,
        "mean_cost": float(np.mean(all_costs)) if all_costs else float("inf"),
        "best_cost": best_cost_b,
        "best_tour": best_tour_b,
        "unique_frac": float(len(seen_keys)) / float(total_candidates),
        "a_mean_cost": float(np.mean(a_costs)) if a_costs else float("nan"),
        "a_best_cost": best_a_cost if a_costs else float("nan"),
        "b_mean_cost": float(np.mean(b_costs)) if b_costs else float("nan"),
        "b_best_cost": best_b_cost if b_costs else float("nan"),
        "det_cost": det_cost_b,
        "det_tour": det_tour_b,
    }


# Module-level lazy process pool for decode parallelization.  Created on first
# use via ``_get_decode_executor`` and shut down at process exit.
_DECODE_EXECUTOR = None
_DECODE_EXECUTOR_WORKERS = 0


def _get_decode_executor(num_workers: int):
    """Return a persistent ProcessPoolExecutor, or None if num_workers<=1."""
    global _DECODE_EXECUTOR, _DECODE_EXECUTOR_WORKERS
    if num_workers is None or num_workers <= 1:
        return None
    if _DECODE_EXECUTOR is not None and _DECODE_EXECUTOR_WORKERS == num_workers:
        return _DECODE_EXECUTOR
    # Worker count changed or first call; (re)create.
    if _DECODE_EXECUTOR is not None:
        try:
            _DECODE_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    import concurrent.futures as _cf
    import multiprocessing as _mp
    # On Linux, fork is fastest (no re-import cost, COW memory).  Workers here
    # never touch CUDA, so fork-after-CUDA-init is safe in our use.  If the
    # user reports deadlocks, switch to "spawn" by overriding the context.
    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context()
    _DECODE_EXECUTOR = _cf.ProcessPoolExecutor(max_workers=int(num_workers), mp_context=ctx)
    _DECODE_EXECUTOR_WORKERS = int(num_workers)
    import atexit as _atexit
    _atexit.register(lambda: _DECODE_EXECUTOR.shutdown(wait=False) if _DECODE_EXECUTOR is not None else None)
    return _DECODE_EXECUTOR


def decode_gumbel_hybrid(
    mu: torch.Tensor,
    C_mod: torch.Tensor,
    cand_mask: torch.Tensor | None,
    D: torch.Tensor,
    root: int = 0,
    twoopt_passes: int = 6,
    num_draws: int = 20,
    gumbel_scale: float = 0.20,
    seed_base: int = 0,
    use_lk_alpha: bool = False,
    noise_type: str = "gumbel",
    tau: float = 0.2,
    seed_split: float = -1.0,
    use_mu_repair: bool = True,
    repair_mode: str = "auto",
    report_timing: bool = False,
    num_workers: int = 1,
) -> Dict[str, np.ndarray | float]:
    r"""Hybrid GPU/CPU decoder. Same interface as decode_gumbel but parallelizes
    noise generation, perturbation, and Prim's MSP on GPU across (M, B) at once.

    Per-instance logic (greedy extraction, repair, cycle extraction, best-of-M
    tracking) remains sequential on CPU but receives precomputed inputs, so the
    heavy compute (Cholesky + Prim's + perturbation) is amortized.

    Returns same dict as decode_gumbel, with an added "_timing" key if
    report_timing=True: {"gpu_s": ..., "cpu_s": ..., "total_s": ...}.
    """
    device = mu.device
    dtype = mu.dtype if mu.is_floating_point() else torch.float32
    B, n = mu.shape[0], mu.shape[-1]
    M = max(1, int(num_draws))

    t0 = time.perf_counter() if report_timing else 0.0

    # ---------- GPU PHASE ----------
    # (1) Batched noise: (M, B, n, n)
    noise_all = _batched_noise_gpu(
        noise_type=noise_type,
        M=M, B=B, n=n, root=root,
        mu_b=mu.to(dtype), C_mod_b=C_mod.to(dtype),
        tau=tau, seed=seed_base, device=device, dtype=dtype,
    )
    # (2) Build Seed B perturbed costs: (M, B, n, n)
    C_mod_perturbed = C_mod.to(dtype).unsqueeze(0) + float(gumbel_scale) * noise_all
    # Symmetrize to guarantee Prim input is symmetric
    C_mod_perturbed = 0.5 * (C_mod_perturbed + C_mod_perturbed.transpose(-1, -2))
    # Zero diagonal
    eye_n = torch.eye(n, device=device, dtype=dtype)
    C_mod_perturbed = C_mod_perturbed * (1.0 - eye_n)

    # (3) Batched Prim's on non-root submatrix per draw
    # Reshape (M, B, n, n) -> (M*B, n, n) and run Prim's on full matrices
    # but we need to mimic "prim on non-root, then add root pair" like map_rooted_onetree_from_cmod.
    # Simpler: run Prim's on full (n, n) excluding root by setting row/col root to inf.
    # Then add root pair (top-2 nearest from root's perspective) afterwards on CPU.
    # Let's do the MSP on non-root subgraph: extract non-root x non-root submatrix.
    nonroot_idx = torch.tensor([i for i in range(n) if i != root], device=device, dtype=torch.long)
    m_size = n - 1
    C_nr = C_mod_perturbed.index_select(-2, nonroot_idx).index_select(-1, nonroot_idx)
    # Flatten (M, B) for Prim
    C_nr_flat = C_nr.reshape(M * B, m_size, m_size)
    adj_nr_flat = _batched_prim_gpu(C_nr_flat, root=0)   # (M*B, m_size, m_size)
    adj_nr = adj_nr_flat.reshape(M, B, m_size, m_size)

    # (4) Root pair: top-2 nearest vertices from root under perturbed cost
    # For each (m, b), argsort C_mod_perturbed[m, b, root, non-root] and take top 2.
    root_costs = C_mod_perturbed[:, :, root, :].index_select(-1, nonroot_idx)  # (M, B, m_size)
    _, root_order = torch.sort(root_costs, dim=-1)    # (M, B, m_size)
    root_pair_local = root_order[..., :2]              # (M, B, 2) — indices into nonroot
    # Map back to full n indices
    root_pair_full = nonroot_idx[root_pair_local]      # (M, B, 2)

    # (5) Seed A scores: log(mu) + noise, precomputed on GPU
    #     Handle mu=0 safely via log(mu + eps)
    EPS_MU = 1e-12
    base_score = torch.log(mu.to(dtype).clamp(min=EPS_MU))  # (B, n, n)
    scores_perturbed = base_score.unsqueeze(0) + float(gumbel_scale) * noise_all
    # Symmetrize scores (matches CPU path)
    scores_perturbed = 0.5 * (scores_perturbed + scores_perturbed.transpose(-1, -2))

    # Transfer to CPU as numpy in one shot
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_end = time.perf_counter() if report_timing else 0.0

    adj_nr_cpu = adj_nr.cpu().numpy().astype(np.int64)     # (M, B, m_size, m_size)
    root_pair_cpu = root_pair_full.cpu().numpy().astype(np.int64)  # (M, B, 2)
    scores_cpu = scores_perturbed.cpu().numpy()             # (M, B, n, n)
    C_mod_cpu = C_mod_perturbed.cpu().numpy()               # (M, B, n, n)
    mu_np = mu.detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)

    # Build full_adj on CPU from non-root adj + root pair edges
    # (need full (n, n) adjacency for repair function)
    full_adj_all = np.zeros((M, B, n, n), dtype=np.int64)
    nonroot_np = np.array([i for i in range(n) if i != root], dtype=np.int64)
    # Scatter non-root adjacency into full matrix
    # adj_nr_cpu is indexed by local non-root indices; map to global
    for mi in range(M):
        for bi in range(B):
            nr = adj_nr_cpu[mi, bi]                # (m_size, m_size)
            rows, cols = np.where(np.triu(nr, k=1) > 0)
            for i, j in zip(rows.tolist(), cols.tolist()):
                u = int(nonroot_np[i])
                v = int(nonroot_np[j])
                full_adj_all[mi, bi, u, v] = 1
                full_adj_all[mi, bi, v, u] = 1
            # Add root pair edges
            a, b = int(root_pair_cpu[mi, bi, 0]), int(root_pair_cpu[mi, bi, 1])
            full_adj_all[mi, bi, root, a] = 1
            full_adj_all[mi, bi, a, root] = 1
            full_adj_all[mi, bi, root, b] = 1
            full_adj_all[mi, bi, b, root] = 1

    # ---------- CPU PHASE ----------
    mean_costs: list[float] = [0.0] * B
    best_costs: list[float] = [0.0] * B
    best_tours: list[list[int]] = [[] for _ in range(B)]
    det_tours: list[list[int]] = [[] for _ in range(B)]
    unique_fracs: list[float] = [0.0] * B
    a_mean_costs: list[float] = [0.0] * B
    a_best_costs: list[float] = [0.0] * B
    b_mean_costs: list[float] = [0.0] * B
    b_best_costs: list[float] = [0.0] * B
    det_costs: list[float] = [0.0] * B

    executor = _get_decode_executor(num_workers) if num_workers and num_workers > 1 else None

    def _per_instance_args(bi: int):
        return dict(
            bi=bi,
            M=M, n=n, root=root,
            twoopt_passes=twoopt_passes,
            seed_split=seed_split,
            use_lk_alpha=use_lk_alpha,
            use_mu_repair=use_mu_repair,
            repair_mode=repair_mode,
            mu_b=mu_np[bi],
            D_b=D_np[bi],
            scores_bi=scores_cpu[:, bi],      # (M, n, n)
            C_mod_bi=C_mod_cpu[:, bi],        # (M, n, n)
            full_adj_bi=full_adj_all[:, bi],  # (M, n, n)
            root_pair_bi=root_pair_cpu[:, bi],# (M, 2)
            cand_bi=(None if cand_np is None else cand_np[bi]),
        )

    def _gather(res: dict):
        bi = int(res["bi"])
        mean_costs[bi] = res["mean_cost"]
        best_costs[bi] = res["best_cost"]
        best_tours[bi] = res["best_tour"]
        unique_fracs[bi] = res["unique_frac"]
        a_mean_costs[bi] = res["a_mean_cost"]
        a_best_costs[bi] = res["a_best_cost"]
        b_mean_costs[bi] = res["b_mean_cost"]
        b_best_costs[bi] = res["b_best_cost"]
        det_costs[bi] = res["det_cost"]
        det_tours[bi] = res["det_tour"]

    if executor is None:
        # Serial path (original behavior).
        for bi in range(B):
            res = _decode_single_instance_cpu(**_per_instance_args(bi))
            _gather(res)
    else:
        # Parallel path via persistent ProcessPoolExecutor.  The slice dict
        # passed per bi is the only payload pickled to workers; we avoid sending
        # the full (M, B, n, n) arrays by indexing ahead of submit().
        futures = [
            executor.submit(_decode_single_instance_cpu, **_per_instance_args(bi))
            for bi in range(B)
        ]
        for fut in futures:
            _gather(fut.result())

    t_total_end = time.perf_counter() if report_timing else 0.0

    result: Dict[str, np.ndarray | float] = {
        "mean_cost": np.array(mean_costs, dtype=np.float64),
        "best_cost": np.array(best_costs, dtype=np.float64),
        "det_cost": np.array(det_costs, dtype=np.float64),
        "best_tour": np.array(best_tours, dtype=object),
        "det_tour": np.array(det_tours, dtype=object),
        "unique_frac": float(np.mean(unique_fracs)),
        "num_draws": M,
        "gumbel_scale": float(gumbel_scale),
        "noise_type": str(noise_type),
        "a_mean_cost": np.array(a_mean_costs, dtype=np.float64),
        "a_best_cost": np.array(a_best_costs, dtype=np.float64),
        "b_mean_cost": np.array(b_mean_costs, dtype=np.float64),
        "b_best_cost": np.array(b_best_costs, dtype=np.float64),
    }
    if report_timing:
        result["_timing"] = {
            "gpu_s": t_gpu_end - t0,
            "cpu_s": t_total_end - t_gpu_end,
            "total_s": t_total_end - t0,
        }
    return result


def decode_gumbel(
    mu: torch.Tensor,
    C_mod: torch.Tensor,
    cand_mask: torch.Tensor | None,
    D: torch.Tensor,
    root: int = 0,
    twoopt_passes: int = 6,
    num_draws: int = 20,
    gumbel_scale: float = 0.20,
    seed_base: int = 0,
    use_lk_alpha: bool = False,
    noise_type: str = "gumbel",
    tau: float = 0.2,
    seed_split: float = -1.0,
    use_mu_repair: bool = True,
    repair_mode: str = "auto",
) -> Dict[str, np.ndarray | float]:
    r"""Dual-seed decoder with configurable noise: mu-greedy + MAP-repair per draw.

    For each of M draws, the same symmetric noise matrix perturbs both paths:
      - score path:  log(mu) + scale * noise  → greedy degree-2 → tour_a  (Seed A)
      - C_mod path:  C_mod   + scale * noise  → MAP 1-tree → μ-weighted repair → tour_b  (Seed B)

    Draw 0 is always deterministic (no noise). All tours optionally LK-improved.

    seed_split controls sample allocation across the two seeds:
      seed_split = -1.0 (default): run BOTH A and B on every draw (legacy). Total tours = 2M.
      seed_split = 0.0: run only Seed B on every draw. Total tours = M.
      seed_split = 1.0: run only Seed A on every draw. Total tours = M.
      seed_split in (0,1): fraction of draws allocated to A; remainder to B.
                           E.g. 0.3 means 30% A, 70% B. Total tours = M.

    use_mu_repair controls whether Seed B uses μ-weighted matching repair (prior-faithful)
    or falls back to backbone+cheapest-insertion repair (legacy, C_mod-based).
    Default True (prior-faithful).
    """
    mu_np = mu.detach().cpu().numpy()
    C_np = C_mod.detach().cpu().numpy()
    cand_np = None if cand_mask is None else cand_mask.detach().cpu().numpy().astype(bool)
    D_np = D.detach().cpu().numpy()
    B, n, _ = D_np.shape
    M = max(1, int(num_draws))

    mean_costs: list[float] = []
    best_costs: list[float] = []
    best_tours: list[list[int]] = []
    det_tours: list[list[int]] = []
    unique_fracs: list[float] = []
    a_mean_costs: list[float] = []
    a_best_costs: list[float] = []
    b_mean_costs: list[float] = []
    b_best_costs: list[float] = []
    det_costs: list[float] = []

    for b in range(B):
        mu_b = mu_np[b]
        D_b = D_np[b]
        C_mod_b = C_np[b]
        C_mod_for_lk = C_mod_b if use_lk_alpha else None

        # Build candidate sets for LK
        candidate_sets: list[set[int]] = []
        if cand_np is not None:
            for i in range(n):
                candidate_sets.append({int(j) for j in range(n) if j != i and cand_np[b, i, j]})
        else:
            full = set(range(n))
            candidate_sets = [full - {i} for i in range(n)]

        rng = np.random.default_rng(int(seed_base + 1000003 * b + 31))
        base_score = _mu_score_matrix(mu_b)

        all_costs: list[float] = []
        a_costs: list[float] = []
        b_costs: list[float] = []
        best_cost_b = float("inf")
        best_tour_b: list[int] = []
        best_a_cost = float("inf")
        best_b_cost = float("inf")
        det_cost_b = float("inf")  # deterministic (no-noise) best of A+B
        det_tour_b: list[int] = []
        seen_keys: set[tuple[int, ...]] = set()

        def _try_seed(seed_tour: list[int]) -> tuple[float, list[int]]:
            nonlocal best_cost_b, best_tour_b
            _, _, lk_tour, lk_cost = _eval_seed(
                seed_tour, D_b, candidate_sets, root, twoopt_passes,
                C_mod_single=C_mod_for_lk,
            )
            all_costs.append(lk_cost)
            seen_keys.add(_tour_key(lk_tour))
            if lk_cost < best_cost_b:
                best_cost_b = lk_cost
                best_tour_b = lk_tour
            return lk_cost, lk_tour

        # Determine which seeds each draw will exercise based on seed_split.
        #   seed_split < 0  → legacy behavior: both A and B every draw
        #   seed_split == 0 → only B
        #   seed_split == 1 → only A
        #   0 < s < 1       → fraction s of draws do A, rest do B. Deterministic
        #                     interleaving (no randomness in allocation).
        run_a_for_draw: list[bool] = [False] * M
        run_b_for_draw: list[bool] = [False] * M
        if seed_split < 0.0:
            for m in range(M):
                run_a_for_draw[m] = True
                run_b_for_draw[m] = True
        else:
            s = max(0.0, min(1.0, float(seed_split)))
            n_a = int(round(s * M))
            # Deterministic interleave: assign the first n_a draws to A, rest to B.
            # Draw 0 (deterministic, no noise) is always run on whichever seed is primary.
            # To preserve the "best-of-both for draw 0" invariant as much as possible:
            # if both seeds get nonzero allocation, run both on draw 0.
            if n_a > 0 and n_a < M:
                run_a_for_draw[0] = True
                run_b_for_draw[0] = True
                # Allocate remaining draws deterministically
                remaining_a = n_a - 1
                remaining_b = M - 1 - remaining_a
                for m in range(1, M):
                    # Alternate based on ratio
                    if remaining_a > 0 and (remaining_b == 0 or
                                            ((m - 1) * n_a) % M < ((m) * n_a) % M):
                        run_a_for_draw[m] = True
                        remaining_a -= 1
                    else:
                        run_b_for_draw[m] = True
                        remaining_b -= 1
            elif n_a == 0:
                for m in range(M):
                    run_b_for_draw[m] = True
            else:  # n_a == M
                for m in range(M):
                    run_a_for_draw[m] = True

        for m in range(M):
            if m == 0:
                noise = np.zeros((n, n), dtype=np.float64)
            else:
                noise = _generate_decode_noise(noise_type, n, root, rng, mu_b=mu_b,
                                               C_mod_b=C_mod_b, tau=tau)

            ca = float("inf")
            cb = float("inf")

            # Seed A: perturbed mu-greedy degree-2
            if run_a_for_draw[m]:
                score_m = base_score + float(gumbel_scale) * noise
                np.fill_diagonal(score_m, -np.inf)
                seed_a, _ = greedy_degree2_tour_from_scores(score_m, D_b, root=root)
                ca, tour_a = _try_seed(seed_a)
                a_costs.append(ca)
                if ca < best_a_cost:
                    best_a_cost = ca

            # Seed B: perturbed MAP 1-tree → (μ-weighted) repair
            if run_b_for_draw[m]:
                C_mod_m = C_mod_b + float(gumbel_scale) * noise
                np.fill_diagonal(C_mod_m, 0.0)
                C_mod_m = 0.5 * (C_mod_m + C_mod_m.T)
                seed_b, _ = map_repair_seed(
                    C_mod_m, D_b, root=root, twoopt_passes=0,
                    mu_single=(mu_b if use_mu_repair else None),
                    repair_mode=repair_mode,
                )
                cb, tour_b = _try_seed(seed_b)
                b_costs.append(cb)
                if cb < best_b_cost:
                    best_b_cost = cb

            # Capture deterministic (no-noise) cost from draw 0
            if m == 0:
                det_cost_b = min(ca, cb)
                if ca <= cb and run_a_for_draw[m]:
                    det_tour_b = tour_a
                elif run_b_for_draw[m]:
                    det_tour_b = tour_b

        total_candidates = max(1, len(all_costs))
        mean_costs.append(float(np.mean(all_costs)))
        best_costs.append(best_cost_b)
        best_tours.append(best_tour_b)
        det_tours.append(det_tour_b)
        unique_fracs.append(float(len(seen_keys)) / float(total_candidates))
        a_mean_costs.append(float(np.mean(a_costs)) if a_costs else float("nan"))
        a_best_costs.append(best_a_cost if a_costs else float("nan"))
        b_mean_costs.append(float(np.mean(b_costs)) if b_costs else float("nan"))
        b_best_costs.append(best_b_cost if b_costs else float("nan"))
        det_costs.append(det_cost_b)

    return {
        "mean_cost": np.array(mean_costs, dtype=np.float64),
        "best_cost": np.array(best_costs, dtype=np.float64),
        "det_cost": np.array(det_costs, dtype=np.float64),
        "best_tour": np.array(best_tours, dtype=object),
        "det_tour": np.array(det_tours, dtype=object),
        "unique_frac": float(np.mean(unique_fracs)),
        "num_draws": M,
        "gumbel_scale": float(gumbel_scale),
        "noise_type": str(noise_type),
        # Per-seed diagnostics
        "a_mean_cost": np.array(a_mean_costs, dtype=np.float64),
        "a_best_cost": np.array(a_best_costs, dtype=np.float64),
        "b_mean_cost": np.array(b_mean_costs, dtype=np.float64),
        "b_best_cost": np.array(b_best_costs, dtype=np.float64),
    }


def nearest_neighbor_cost(D: torch.Tensor) -> np.ndarray:
    D_np = D.detach().cpu().numpy()
    B, n, _ = D_np.shape
    costs = []
    for b in range(B):
        best = float("inf")
        for start in range(min(n, 5)):
            vis = {start}
            tc, cur = 0.0, start
            for _ in range(n - 1):
                row = D_np[b, cur].copy()
                row[list(vis)] = float("inf")
                nxt = row.argmin()
                tc += row[nxt]
                vis.add(nxt)
                cur = nxt
            tc += D_np[b, cur, start]
            best = min(best, tc)
        costs.append(best)
    return np.array(costs)
