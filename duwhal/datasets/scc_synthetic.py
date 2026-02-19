"""
demo/synthetic.py — Controlled 3-SCC synthetic dataset.
Generates transactions with known sink SCCs for testing.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def generate_3scc_dataset(
    nodes_per_scc: int = 20,
    n_transient: int = 10,
    baskets_per_scc: int = 500,
    bridge_baskets: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate a synthetic transaction dataset with exactly 3 sink SCCs,
    some transient bridge nodes, and held-out test baskets.
    
    Structure:
    - SCC-0: nodes 0..19 (e.g. "retro" product cluster)
    - SCC-1: nodes 20..39 (e.g. "modern" product cluster)
    - SCC-2: nodes 40..59 (e.g. "indie" product cluster)
    - Transient: nodes 60..69 (bridge products that point into SCCs)
    """
    rng = np.random.default_rng(seed)
    
    all_baskets = []
    basket_id = 0
    
    # Ground truth
    scc_ranges = [
        list(range(0, nodes_per_scc)),
        list(range(nodes_per_scc, 2 * nodes_per_scc)),
        list(range(2 * nodes_per_scc, 3 * nodes_per_scc)),
    ]
    transient_range = list(range(3 * nodes_per_scc, 3 * nodes_per_scc + n_transient))
    
    # 1. Dense intra-SCC baskets — creates strong internal connectivity
    for scc_idx, scc_nodes in enumerate(scc_ranges):
        for _ in range(baskets_per_scc):
            # Pick 2-4 items from same SCC
            k = rng.integers(2, 5)
            items = rng.choice(scc_nodes, size=min(k, len(scc_nodes)), replace=False)
            for item in items:
                all_baskets.append((f"B{basket_id}", f"P{item}"))
            basket_id += 1
    
    # 2. Bridge baskets: transient nodes co-occur with specific SCC nodes
    #    Each transient node appears in ~20 baskets with 2-3 specific SCC targets.
    #    The SCC nodes appear in ~150+ baskets total, so:
    #      p(transient → SCC_target) = 20/20 = 1.0 (high)
    #      p(SCC_target → transient) = 20/150 = 0.13 (low, below confidence threshold)
    #    This creates the directional asymmetry that makes them transient.
    for t_node in transient_range:
        target_scc = rng.integers(0, 3)
        # Pick 2-3 *specific* SCC nodes to pair with
        scc_targets = rng.choice(scc_ranges[target_scc], size=3, replace=False)
        for _ in range(20):
            target_item = rng.choice(scc_targets)
            all_baskets.append((f"B{basket_id}", f"P{t_node}"))
            all_baskets.append((f"B{basket_id}", f"P{target_item}"))
            basket_id += 1
    
    # 3. A few cross-SCC baskets (noise) — should be filtered by min_support
    for _ in range(10):
        s1 = rng.integers(0, 3)
        s2 = (s1 + 1) % 3
        i1 = rng.choice(scc_ranges[s1])
        i2 = rng.choice(scc_ranges[s2])
        all_baskets.append((f"B{basket_id}", f"P{i1}"))
        all_baskets.append((f"B{basket_id}", f"P{i2}"))
        basket_id += 1
    
    df = pd.DataFrame(all_baskets, columns=["basket_id", "product_id"])
    
    # Generate held-out test baskets
    test_baskets = []
    for scc_idx, scc_nodes in enumerate(scc_ranges):
        for _ in range(50):
            k = rng.integers(2, 5)
            items = rng.choice(scc_nodes, size=min(k, len(scc_nodes)), replace=False)
            test_baskets.append([int(i) for i in items])
    
    metadata = {
        "scc_ranges": scc_ranges,
        "transient_range": transient_range,
        "n_nodes": 3 * nodes_per_scc + n_transient,
        "n_baskets": basket_id,
        "n_sccs_expected": 3,
        "test_baskets": test_baskets,
        "node_labels": {i: f"P{i}" for i in range(3 * nodes_per_scc + n_transient)},
    }
    
    return df, metadata
