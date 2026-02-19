"""
duwhal.datasets.scaling — Large-scale synthetic data generators.

Datasets for performance benchmarking, ingestion testing, and scalability analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def generate_large_scale_data(
    n_transactions: int = 100_000,
    n_items: int = 1_000,
    density: int = 8,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a large transaction dataset with power-law item distribution.

    80% of transactions involve the top 20% of items, creating a realistic
    long-tail distribution suitable for stress testing ingestion, mining,
    and recommendation flows.

    Columns: ``order_id``, ``item_id``

    Parameters
    ----------
    n_transactions : int
        Number of distinct orders.
    n_items : int
        Size of the item catalog.
    density : int
        Average items per order.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        n_transactions × density rows.

    Example
    -------
    >>> from duwhal.datasets import generate_large_scale_data
    >>> df = generate_large_scale_data(n_transactions=1000, n_items=100)
    >>> df.shape
    (8000, 2)
    """
    rng = np.random.default_rng(seed)

    # Power-law: 80% of transactions hit top 20% of items
    n_popular = max(1, int(n_items * 0.2))
    p_popular = 0.8 / n_popular
    p_other = 0.2 / max(1, n_items - n_popular)
    probs = np.array([p_popular] * n_popular + [p_other] * (n_items - n_popular))
    probs /= probs.sum()

    order_ids = np.repeat(np.arange(n_transactions), density)
    item_indices = rng.choice(n_items, size=len(order_ids), p=probs)

    return pd.DataFrame({
        "order_id": [f"T{oid}" for oid in order_ids],
        "item_id": [f"I{iid}" for iid in item_indices],
    })
