"""
graph/scc.py — Sink SCC identification + spectral validation + condensation DAG.
Wraps duwhal.SinkSCCFinder for the algorithmic heavy-lifting, adds spectral
cross-validation and hierarchical decomposition.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from typing import Dict, List, Set, Tuple, Optional


def find_sink_sccs(
    P: sparse.csr_matrix,
    reverse_index: Dict[int, str],
    min_edge_weight: float = 0.0,
) -> Dict:
    """
    Find all SCCs via Tarjan's, identify sinks, build condensation DAG.
    Operates on the scipy sparse matrix P directly (index-based).
    
    Parameters
    ----------
    min_edge_weight : float
        Minimum transition probability to keep an edge. Edges below this
        threshold are removed before SCC detection (Step 6 from spec:
        threshold weak edges to make sink identification robust to noise).
    """
    n = P.shape[0]
    adj = _sparse_to_adj(P, min_edge_weight)

    sccs = _tarjan(adj, n)
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for v in scc:
            node_to_scc[v] = i

    sink_flags = _identify_sinks(adj, sccs, node_to_scc)
    sink_sccs = [sccs[i] for i in range(len(sccs)) if sink_flags[i]]
    transient_nodes = [v for v in range(n) if node_to_scc.get(v) is not None and not sink_flags[node_to_scc[v]]]
    # Also include nodes not in any SCC (isolates with no outgoing edges may be trivial sinks)

    condensation = _build_condensation(adj, sccs, node_to_scc)

    # Spectral validation + stationary distributions
    stationary = {}
    for idx, scc in enumerate(sink_sccs):
        sub_P = _extract_submatrix(P, scc)
        stat_dist = _compute_stationary(sub_P)
        stationary[idx] = {
            "nodes": [reverse_index[v] for v in scc],
            "stationary_dist": stat_dist,
            "eigenvalue_1_count": _count_unit_eigenvalues(sub_P),
        }

    return {
        "sccs": [[reverse_index[v] for v in scc] for scc in sccs],
        "sink_sccs": [[reverse_index[v] for v in scc] for scc in sink_sccs],
        "sink_sccs_idx": sink_sccs,
        "transient_nodes": [reverse_index[v] for v in transient_nodes],
        "condensation_edges": condensation,
        "stationary": stationary,
        "n_sinks": len(sink_sccs),
        "n_sccs": len(sccs),
    }


def _sparse_to_adj(P: sparse.csr_matrix, min_weight: float = 0.0) -> Dict[int, List[int]]:
    """Convert sparse matrix to adjacency list, filtering weak edges."""
    adj: Dict[int, List[int]] = {}
    coo = P.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if w >= min_weight:
            adj.setdefault(int(i), []).append(int(j))
    return adj


def _tarjan(adj: Dict[int, List[int]], n: int) -> List[List[int]]:
    """Tarjan's SCC algorithm — iterative to avoid stack overflow on large graphs."""
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

    # Iterative version using explicit call stack
    for start in range(n):
        if start in index:
            continue
        call_stack = [(start, 0)]  # (node, neighbor_index)
        while call_stack:
            v, ni = call_stack[-1]
            if v not in index:
                index[v] = lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack.add(v)

            neighbors = adj.get(v, [])
            if ni < len(neighbors):
                w = neighbors[ni]
                call_stack[-1] = (v, ni + 1)
                if w not in index:
                    call_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                # All neighbors processed
                if lowlink[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)

                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

    return sccs


def _identify_sinks(
    adj: Dict[int, List[int]],
    sccs: List[List[int]],
    node_to_scc: Dict[int, int],
) -> List[bool]:
    """A sink SCC has no outgoing edges to other SCCs."""
    is_sink = [True] * len(sccs)
    for i, scc in enumerate(sccs):
        scc_set = set(scc)
        for v in scc:
            for w in adj.get(v, []):
                if w not in scc_set:
                    is_sink[i] = False
                    break
            if not is_sink[i]:
                break
    return is_sink


def _build_condensation(
    adj: Dict[int, List[int]],
    sccs: List[List[int]],
    node_to_scc: Dict[int, int],
) -> List[Tuple[int, int]]:
    """Build edges of the condensation DAG."""
    edges = set()
    for i, scc in enumerate(sccs):
        for v in scc:
            for w in adj.get(v, []):
                j = node_to_scc.get(w)
                if j is not None and j != i:
                    edges.add((i, j))
    return sorted(edges)


def _extract_submatrix(P: sparse.csr_matrix, nodes: List[int]) -> np.ndarray:
    """Extract and renormalize the submatrix for an SCC."""
    idx = sorted(nodes)
    sub = P[np.ix_(idx, idx)].toarray()
    # Renormalize rows
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return sub / row_sums


def _compute_stationary(sub_P: np.ndarray) -> np.ndarray:
    """Compute stationary distribution via left eigenvector of unit eigenvalue."""
    n = sub_P.shape[0]
    if n <= 1:
        return np.ones(n)
    try:
        # Left eigenvector: pi @ P = pi, equiv to P^T @ pi = pi
        vals, vecs = np.linalg.eig(sub_P.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi
    except Exception:
        return np.ones(n) / n


def _count_unit_eigenvalues(sub_P: np.ndarray, tol: float = 1e-6) -> int:
    """Count eigenvalues within tol of 1.0 — should be exactly 1 for irreducible chain."""
    vals = np.linalg.eigvals(sub_P)
    return int(np.sum(np.abs(vals - 1.0) < tol))


def hierarchical_decompose(
    P: sparse.csr_matrix,
    sink_nodes: List[int],
    depth: int = 3,
    base_threshold: float = 0.01,
) -> Dict:
    """
    Recursively decompose a sink SCC into sub-communities
    by progressively raising the edge threshold.
    """
    tree = {"nodes": sink_nodes, "children": []}
    if depth <= 0 or len(sink_nodes) <= 2:
        return tree

    sub_P = _extract_submatrix(P, sink_nodes)
    # Raise threshold to reveal sub-structure
    threshold = base_threshold * 2
    mask = sub_P >= threshold
    sub_P_filtered = sub_P * mask

    # Re-normalize
    row_sums = sub_P_filtered.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    sub_P_filtered /= row_sums

    # Find SCCs in filtered subgraph
    n = len(sink_nodes)
    adj = {}
    for i in range(n):
        for j in range(n):
            if sub_P_filtered[i, j] > 0:
                adj.setdefault(i, []).append(j)

    sub_sccs = _tarjan(adj, n)
    if len(sub_sccs) <= 1:
        return tree

    for scc in sub_sccs:
        child_nodes = [sink_nodes[i] for i in scc]
        child = hierarchical_decompose(P, child_nodes, depth - 1, threshold)
        tree["children"].append(child)

    return tree
