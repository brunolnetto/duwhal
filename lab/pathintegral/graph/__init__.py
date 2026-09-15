"""
graph/builder.py — Builds the directed weighted transition matrix P from duwhal interactions.
Delegates co-occurrence counting to DuckDB, then exports as scipy sparse.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Dict, Tuple, Optional
from duwhal import Duwhal


def build_transition_matrix(
    db: Duwhal,
    min_support: int = 5,
    alpha: float = 0.1,
) -> Dict:
    """
    Build directed transition matrix P from loaded interactions.
    
    P_ij = (count(i,j) + alpha) / (count(i) + alpha * |V|)
    
    Edges below min_support are zeroed before normalization.
    """
    # Extract co-occurrence counts and node totals via DuckDB
    edges_raw = db.sql(f"""
        SELECT a.node_id AS source, b.node_id AS target, COUNT(*) AS cooc
        FROM {db.table_name} a
        JOIN {db.table_name} b ON a.set_id = b.set_id AND a.node_id != b.node_id
        GROUP BY 1, 2
    """).to_pylist()

    totals_raw = db.sql(f"""
        SELECT node_id, COUNT(*) AS total
        FROM {db.table_name}
        GROUP BY 1
    """).to_pylist()

    # Build index
    nodes = sorted({r["source"] for r in edges_raw} | {r["target"] for r in edges_raw} | {r["node_id"] for r in totals_raw})
    index = {n: i for i, n in enumerate(nodes)}
    reverse_index = {i: n for n, i in index.items()}
    n = len(nodes)

    # Build sparse count matrix
    rows, cols, data = [], [], []
    for r in edges_raw:
        if r["cooc"] >= min_support:
            rows.append(index[r["source"]])
            cols.append(index[r["target"]])
            data.append(float(r["cooc"]))

    C = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Build total vector
    total_vec = np.zeros(n)
    for r in totals_raw:
        if r["node_id"] in index:
            total_vec[index[r["node_id"]]] = float(r["total"])

    # Laplace smoothing + row normalization -> P
    P = _normalize_with_smoothing(C, total_vec, alpha, n)

    return {
        "P": P,
        "index": index,
        "reverse_index": reverse_index,
        "n_nodes": n,
    }


def _normalize_with_smoothing(
    C: sparse.csr_matrix,
    total_vec: np.ndarray,
    alpha: float,
    n: int,
) -> sparse.csr_matrix:
    """Row-normalize count matrix with Dirichlet smoothing."""
    rows, cols, vals = [], [], []
    C_dense = C.toarray()  # For moderate catalogs; for huge ones use sparse ops
    
    for i in range(n):
        row = C_dense[i]
        # Only add alpha to observed (nonzero) entries to maintain sparsity
        nonzero_mask = row > 0
        if not np.any(nonzero_mask):
            continue
        smoothed = row.copy()
        smoothed[nonzero_mask] += alpha
        # Zero diagonal
        smoothed[i] = 0.0
        row_sum = smoothed.sum()
        if row_sum > 0:
            smoothed /= row_sum
            for j in np.nonzero(smoothed)[0]:
                rows.append(i)
                cols.append(j)
                vals.append(smoothed[j])

    P = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return P


def validate_transition_matrix(P: sparse.csr_matrix, tol: float = 1e-6) -> Dict:
    """Validate structural properties of the transition matrix."""
    n = P.shape[0]
    row_sums = np.array(P.sum(axis=1)).flatten()
    
    # Only check rows that have any edges
    active_rows = row_sums > tol
    
    checks = {
        "rows_sum_to_1": bool(np.allclose(row_sums[active_rows], 1.0, atol=tol)),
        "no_negative": bool((P.data >= 0).all()) if len(P.data) > 0 else True,
        "zero_diagonal": bool(np.allclose(P.diagonal(), 0.0, atol=tol)),
        "density": P.nnz / (n * n) if n > 0 else 0.0,
        "n_active_rows": int(active_rows.sum()),
        "n_nodes": n,
    }
    return checks
