"""
evaluation/structural.py — Structural tests for path-integral properties.
These tests verify mathematical consistency, independent of recommendation accuracy.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
from examples.pathintegral.propagator import compute_propagator, propagator_sweep
from examples.pathintegral.graph.scc import find_sink_sccs


def test_chapman_kolmogorov(
    P: sparse.csr_matrix,
    n_triples: int = 100,
    tol: float = 1e-4,
    rng: np.random.Generator = None,
) -> Dict:
    """
    Chapman-Kolmogorov consistency test:
    K(A, B; T1+T2) = Σ_C K(A, C; T1) * K(C, B; T2)
    
    Tests whether the propagator respects the Markov semigroup property.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = P.shape[0]
    T1, T2 = 1.0, 1.5

    K_T1 = compute_propagator(P, T1)["K"]
    K_T2 = compute_propagator(P, T2)["K"]
    K_sum = compute_propagator(P, T1 + T2)["K"]

    # K(T1+T2) should equal K(T1) @ K(T2) (matrix product = sum over intermediates)
    K_composed = K_T1 @ K_T2

    passes = 0
    failures = []
    for _ in range(n_triples):
        A = rng.integers(0, n)
        B = rng.integers(0, n)
        direct = K_sum[A, B]
        composed = K_composed[A, B]
        if abs(direct - composed) < tol:
            passes += 1
        else:
            failures.append({"A": int(A), "B": int(B), "direct": float(direct), "composed": float(composed), "diff": float(abs(direct - composed))})

    return {
        "test": "Chapman-Kolmogorov",
        "n_triples": n_triples,
        "passes": passes,
        "pass_rate": passes / n_triples,
        "T1": T1,
        "T2": T2,
        "failures": failures[:5],
    }


def test_sink_absorption(
    P: sparse.csr_matrix,
    reverse_index: Dict[int, str],
    T_values: List[float] = None,
    min_edge_weight: float = 0.0,
) -> Dict:
    """
    Sink SCC absorption test: for each transient node A and each sink S,
    verify that Σ_{B∈S} K(A, B; T) is monotonically increasing in T.
    """
    if T_values is None:
        T_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    scc_result = find_sink_sccs(P, reverse_index, min_edge_weight=min_edge_weight)
    sink_sccs_idx = scc_result["sink_sccs_idx"]

    if not sink_sccs_idx:
        return {"test": "sink_absorption", "status": "no_sinks_found", "passed": True}

    # Find transient nodes (not in any sink)
    all_sink_nodes = set()
    for scc in sink_sccs_idx:
        all_sink_nodes.update(scc)
    transient = [i for i in range(P.shape[0]) if i not in all_sink_nodes]

    if not transient:
        return {"test": "sink_absorption", "status": "no_transient_nodes", "passed": True}

    K_cache = propagator_sweep(P, T_values)

    violations = []
    tests_run = 0
    for a in transient[:10]:  # Sample up to 10 transient nodes
        for s_idx, scc in enumerate(sink_sccs_idx):
            prev_mass = 0.0
            monotonic = True
            masses = []
            for T in T_values:
                K = K_cache[T]
                mass = sum(K[a, b] for b in scc)
                masses.append(float(mass))
                if mass < prev_mass - 1e-6:
                    monotonic = False
                prev_mass = mass

            tests_run += 1
            if not monotonic:
                violations.append({
                    "transient_node": reverse_index[a],
                    "sink_scc": [reverse_index[v] for v in scc],
                    "masses": masses,
                })

    return {
        "test": "sink_absorption",
        "passed": len(violations) == 0,
        "tests_run": tests_run,
        "violations": violations,
        "T_values": T_values,
    }


def test_spectral_gap_alignment(
    P: sparse.csr_matrix,
    reverse_index: Dict[int, str],
    tol: float = 1e-6,
    min_edge_weight: float = 0.0,
) -> Dict:
    """
    Verify: # sink SCCs found by Tarjan == # eigenvalues of P within tol of 1.0.
    Cross-validates graph algorithm against spectral condition.
    """
    scc_result = find_sink_sccs(P, reverse_index, min_edge_weight=min_edge_weight)
    n_sinks = scc_result["n_sinks"]

    eigenvalues = np.linalg.eigvals(P.toarray())
    n_unit_eigs = int(np.sum(np.abs(eigenvalues - 1.0) < tol))

    return {
        "test": "spectral_gap_alignment",
        "n_sink_sccs": n_sinks,
        "n_unit_eigenvalues": n_unit_eigs,
        "aligned": n_sinks == n_unit_eigs,
        "eigenvalue_magnitudes": sorted(np.abs(eigenvalues).tolist(), reverse=True)[:10],
    }


def test_degradation_under_sampling(
    P: sparse.csr_matrix,
    T: float = 5.0,
    fractions: List[float] = None,
    rng: np.random.Generator = None,
) -> Dict:
    """
    Subsample edges and measure how gracefully K(T) degrades.
    """
    if fractions is None:
        fractions = [1.0, 0.5, 0.25, 0.1]
    if rng is None:
        rng = np.random.default_rng(42)

    K_full = compute_propagator(P, T)["K"]
    norm_full = np.linalg.norm(K_full, "fro")

    results = []
    for frac in fractions:
        P_sub = _subsample_edges(P, frac, rng)
        K_sub = compute_propagator(P_sub, T)["K"]
        dist = np.linalg.norm(K_full - K_sub, "fro")
        results.append({
            "fraction": frac,
            "frobenius_distance": float(dist),
            "relative_distance": float(dist / norm_full) if norm_full > 0 else 0.0,
        })

    return {
        "test": "degradation_under_sampling",
        "T": T,
        "results": results,
    }


def _subsample_edges(
    P: sparse.csr_matrix,
    fraction: float,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Randomly zero out (1-fraction) of edges, then renormalize."""
    if fraction >= 1.0:
        return P.copy()

    coo = P.tocoo()
    mask = rng.random(len(coo.data)) < fraction
    data = coo.data * mask
    P_sub = sparse.csr_matrix((data, (coo.row, coo.col)), shape=P.shape)

    # Renormalize rows
    row_sums = np.array(P_sub.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums)
    return inv @ P_sub


def run_all_structural_tests(
    P: sparse.csr_matrix,
    reverse_index: Dict[int, str],
    min_edge_weight: float = 0.0,
) -> Dict:
    """Run the full structural test suite."""
    results = {}
    results["chapman_kolmogorov"] = test_chapman_kolmogorov(P)
    results["sink_absorption"] = test_sink_absorption(P, reverse_index, min_edge_weight=min_edge_weight)
    results["spectral_gap"] = test_spectral_gap_alignment(P, reverse_index, min_edge_weight=min_edge_weight)
    results["sampling_degradation"] = test_degradation_under_sampling(P)

    all_passed = all([
        results["chapman_kolmogorov"]["pass_rate"] > 0.99,
        results["sink_absorption"]["passed"],
        results["spectral_gap"]["aligned"],
    ])
    results["all_passed"] = all_passed
    return results
