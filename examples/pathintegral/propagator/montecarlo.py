"""
propagator/montecarlo.py — MCMC path sampler for the propagator.
Estimates K(A, B; T) by sampling random walk trajectories and recording paths.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def sample_propagator(
    P: sparse.csr_matrix,
    seed_idx: int,
    T: float,
    n_samples: int = 10000,
    reverse_index: Optional[Dict[int, str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Estimate K(A, ·; T) by sampling random walk trajectories from seed_idx.
    
    Walk length is drawn from Poisson(T) — continuous-time Markov chain analog.
    K(A, B; T) is estimated as fraction of walks visiting B.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = P.shape[0]
    P_dense = P.toarray()

    visit_counts = np.zeros(n)
    paths_by_target: Dict[int, List[List[int]]] = defaultdict(list)

    for _ in range(n_samples):
        walk_length = rng.poisson(T)
        walk_length = max(walk_length, 1)

        path = [seed_idx]
        current = seed_idx
        for _ in range(walk_length):
            row = P_dense[current]
            row_sum = row.sum()
            if row_sum < 1e-12:
                break  # Absorbing node
            prob = row / row_sum
            current = rng.choice(n, p=prob)
            path.append(current)

        # Record visits (unique per walk)
        visited = set(path)
        for v in visited:
            visit_counts[v] += 1

        # Store path for the terminal node
        terminal = path[-1]
        if len(paths_by_target[terminal]) < 5:  # Keep top 5 paths
            paths_by_target[terminal].append(path)

    scores = visit_counts / n_samples

    # Build output
    result_scores = {}
    result_paths = {}
    for idx in range(n):
        label = reverse_index[idx] if reverse_index else str(idx)
        if scores[idx] > 0:
            result_scores[label] = float(scores[idx])
            if idx in paths_by_target:
                if reverse_index:
                    result_paths[label] = [
                        [reverse_index[v] for v in p] for p in paths_by_target[idx][:3]
                    ]
                else:
                    result_paths[label] = paths_by_target[idx][:3]

    return {
        "scores": result_scores,
        "paths": result_paths,
        "n_samples": n_samples,
        "T": T,
    }
