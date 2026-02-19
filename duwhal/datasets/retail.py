"""
duwhal.datasets.retail — Retail / E-commerce transaction generators.

Datasets for market basket analysis, cross-sell/upsell patterns, and
recommendation benchmarking.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional


def generate_retail_transactions(seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic e-commerce transactions with known patterns.

    Contains three product categories with deliberate co-occurrence signals:
    - **Electronics**: iPhone → Silicone Case (complementary accessory)
    - **Grocery**: Pasta → Tomato Sauce (cross-category staple)
    - **Breakfast**: Whole Milk → Oat Cereal (common bundle)

    Columns: ``order_id``, ``item_name``, ``timestamp``

    Returns
    -------
    pd.DataFrame
        A DataFrame with 17 rows across 8 orders.

    Example
    -------
    >>> from duwhal.datasets import generate_retail_transactions
    >>> df = generate_retail_transactions()
    >>> df.head()
    """
    transactions = [
        # Electronics
        ("O1", "iPhone 15", "2024-01-01 10:00:00"),
        ("O1", "Silicone Case", "2024-01-01 10:00:05"),
        ("O2", "iPhone 15", "2024-01-01 11:00:00"),
        ("O2", "Silicone Case", "2024-01-01 11:00:10"),
        ("O3", "iPhone 15", "2024-01-01 12:00:00"),
        ("O3", "Screen Protector", "2024-01-01 12:00:15"),
        # Grocery
        ("O4", "Pasta", "2024-01-02 09:00:00"),
        ("O4", "Tomato Sauce", "2024-01-02 09:00:05"),
        ("O5", "Pasta", "2024-01-02 10:00:00"),
        ("O5", "Tomato Sauce", "2024-01-02 10:00:10"),
        ("O5", "Parmesan", "2024-01-02 10:00:20"),
        # Breakfast
        ("O6", "Whole Milk", "2024-01-03 08:00:00"),
        ("O6", "Oat Cereal", "2024-01-03 08:00:10"),
        ("O7", "Whole Milk", "2024-01-03 09:00:00"),
        ("O7", "Oat Cereal", "2024-01-03 09:00:15"),
        ("O8", "Oat Cereal", "2024-01-03 10:00:00"),
        ("O8", "Banana", "2024-01-03 10:00:05"),
    ]
    return pd.DataFrame(transactions, columns=["order_id", "item_name", "timestamp"])


def generate_benchmark_patterns(seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate transactions with strong, deliberate patterns for benchmarking.

    Three clear signals:
    - {Beer, Diaper} — 100% co-occurrence (classic apocryphal pattern)
    - {Milk, Bread, Butter} — frequent triple
    - {Coke} — popular singleton (baseline noise)

    Columns: ``order_id``, ``item_id``

    Returns
    -------
    pd.DataFrame
        ~550 rows with known ground-truth patterns.
    """
    # Pattern 1: {Beer, Diaper} co-occur 100%
    p1 = [
        ["T1_a", "Beer"], ["T1_a", "Diaper"],
        ["T1_b", "Beer"], ["T1_b", "Diaper"],
    ] * 50

    # Pattern 2: Sequence A -> B -> C
    p2 = []
    for i in range(50):
        p2.extend([
            [f"T2_{i}", "Milk"],
            [f"T2_{i}", "Bread"],
            [f"T2_{i}", "Butter"],
        ])

    # Pattern 3: Popular item (Coke)
    p3 = [[f"T3_{i}", "Coke"] for i in range(200)]

    return pd.DataFrame(p1 + p2 + p3, columns=["order_id", "item_id"])
