"""
duwhal.profiling — Rule and Item profile clustering.

Both profilers work with the raw metric space produced by
``Duwhal.association_rules()`` without requiring hard-coded labels,
domain knowledge, or a fixed number of clusters.
"""

from .item_profiler import ItemCluster, ItemProfiler
from .rule_profiler import RuleCluster, RuleProfiler

__all__ = [
    "RuleCluster",
    "RuleProfiler",
    "ItemCluster",
    "ItemProfiler",
]
