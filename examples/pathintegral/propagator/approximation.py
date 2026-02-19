"""
propagator/approximation.py — Rank-r truncated SVD approximation.
Core module for proving that classical CF is a degenerate limit 
of the path-integral propagator at T → ∞ and r → |V|.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import Dict, List, Optional


def compute_rank_r_propagator(
    P: sparse.csr_matrix,
    T: float,
    r: int,
) -> np.ndarray:
    """
    Compute rank-r truncated SVD approximation of the propagator.
    
    1. Symmetrize: P_tilde = D^{-1/2} P D^{1/2}
    2. Truncated SVD: P_tilde_r = U_r Sigma_r V_r^T
    3. Propagator: K_r(T) = U_r diag(exp((sigma_k - 1) * T)) V_r^T
    """
    n = P.shape[0]
    r = min(r, n - 1)  # svds requires k < min(m, n)
    if r < 1:
        return np.zeros((n, n))

    # Degree vector
    D_vec = np.array(P.sum(axis=1)).flatten()
    D_vec[D_vec == 0] = 1.0

    # Symmetrization: D^{-1/2} P D^{1/2}
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(D_vec))
    D_sqrt = sparse.diags(np.sqrt(D_vec))
    P_sym = D_inv_sqrt @ P @ D_sqrt

    # Truncated SVD
    try:
        U, sigma, Vt = svds(P_sym.astype(float), k=r)
    except Exception:
        # Fallback for very small matrices
        U_full, sigma_full, Vt_full = np.linalg.svd(P_sym.toarray())
        U = U_full[:, :r]
        sigma = sigma_full[:r]
        Vt = Vt_full[:r, :]

    # Propagator in symmetrized basis
    exp_diag = np.exp((sigma - 1.0) * T)
    K_r_sym = U @ np.diag(exp_diag) @ Vt

    # Un-symmetrize: K_r = D^{1/2} K_r_sym D^{-1/2}
    K_r = D_sqrt.toarray() @ K_r_sym @ D_inv_sqrt.toarray()
    return np.clip(np.real(K_r), 0, None)


def residual_sweep(
    P: sparse.csr_matrix,
    K_exact_cache: Dict[float, np.ndarray],
    T_values: List[float],
    r_values: List[int],
) -> Dict:
    """
    Compute residual ||K(T) - K_r(T)||_F for all (T, r) pairs.
    
    Parameters
    ----------
    P : transition matrix
    K_exact_cache : precomputed exact propagators {T: K(T)}
    T_values : list of diffusion times
    r_values : list of ranks to sweep
    
    Returns
    -------
    dict with 'residuals' (|T| x |r| array) and 'relative_residuals'
    """
    residuals = np.zeros((len(T_values), len(r_values)))
    relative = np.zeros_like(residuals)

    for ti, T in enumerate(T_values):
        K_exact = K_exact_cache[T]
        norm_exact = np.linalg.norm(K_exact, "fro")
        for ri, r in enumerate(r_values):
            K_r = compute_rank_r_propagator(P, T, r)
            res = np.linalg.norm(K_exact - K_r, "fro")
            residuals[ti, ri] = res
            relative[ti, ri] = res / norm_exact if norm_exact > 0 else 0.0

    return {
        "residuals": residuals,
        "relative_residuals": relative,
        "T_values": T_values,
        "r_values": r_values,
    }


def basket_completion_accuracy(
    K: np.ndarray,
    baskets: List[List[int]],
    n_rec: int = 10,
) -> float:
    """
    Evaluate basket completion: for each basket, use first item to predict rest.
    Returns hit rate at n_rec.
    """
    if not baskets:
        return 0.0
    hits = 0
    total = 0
    for basket in baskets:
        if len(basket) < 2:
            continue
        seed = basket[0]
        targets = set(basket[1:])
        scores = K[seed].copy()
        scores[seed] = -1.0
        top_n = set(np.argsort(scores)[::-1][:n_rec])
        if top_n & targets:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0
