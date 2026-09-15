"""
propagator/exact.py — Exact matrix exponentiation propagator.
K(T) = expm(L * T) where L = P - I is the graph Laplacian.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from typing import Dict, List, Tuple, Optional


def compute_propagator(
    P: sparse.csr_matrix,
    T: float,
    reverse_index: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Compute the exact propagator K(T) = expm((P - I) * T).
    
    Parameters
    ----------
    P : transition matrix (n x n)
    T : diffusion time parameter
    reverse_index : optional mapping from index to product name
    """
    n = P.shape[0]
    L = P.toarray() - np.eye(n)
    K = expm(L * T)
    return {"K": K, "T": T, "L": L}


def recommend_from_propagator(
    K: np.ndarray,
    seed_idx: int,
    reverse_index: Dict[int, str],
    n: int = 10,
    exclude_seed: bool = True,
) -> List[Tuple[str, float]]:
    """Return top-n items from row of propagator, sorted by score."""
    scores = K[seed_idx].copy()
    if exclude_seed:
        scores[seed_idx] = -1.0
    top_idx = np.argsort(scores)[::-1][:n]
    return [(reverse_index[i], float(scores[i])) for i in top_idx]


def propagator_sweep(
    P: sparse.csr_matrix,
    T_values: List[float],
    seed_idx: Optional[int] = None,
) -> Dict:
    """
    Efficiently compute K(T) for multiple T values.
    Uses eigendecomposition: K(T) = V diag(e^{(λ-1)T}) V^{-1}.
    """
    n = P.shape[0]
    L = P.toarray() - np.eye(n)

    # Eigendecomposition of L (cached, expensive step done once)
    eigenvalues, V = np.linalg.eig(L)
    V_inv = np.linalg.inv(V)

    results = {}
    for T in T_values:
        D = np.diag(np.exp(eigenvalues * T))
        K = np.real(V @ D @ V_inv)
        # Clip small negatives from floating point
        K = np.clip(K, 0, None)
        results[T] = K

    return results
