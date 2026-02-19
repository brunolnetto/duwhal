"""
Evaluation metrics for discovery and recommendation.
"""

from __future__ import annotations
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pyarrow as pa

def _get_hits_at_k(recommended: List[Any], actual: List[Any], k: int) -> int:
    rec_k = recommended[:k]
    if not rec_k: return 0
    actual_set = set(actual)
    return len([x for x in rec_k if x in actual_set])

def precision_at_k(recommended: List[Any], actual: List[Any], k: int) -> float:
    if k <= 0: return 0.0
    return _get_hits_at_k(recommended, actual, k) / k

def recall_at_k(recommended: List[Any], actual: List[Any], k: int) -> float:
    if not actual: return 0.0
    hits = _get_hits_at_k(recommended, actual, k)
    return hits / len(actual)

def f1_at_k(recommended: List[Any], actual: List[Any], k: int) -> float:
    p = precision_at_k(recommended, actual, k)
    r = recall_at_k(recommended, actual, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def average_precision(recommended: List[Any], actual: List[Any]) -> float:
    if not actual: return 0.0
    actual_set, score, num_hits = set(actual), 0.0, 0
    for i, item in enumerate(recommended):
        if item in actual_set:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(actual)

def _calculate_dcg(items: List[Any], actual_set: set) -> float:
    return sum(1.0 / np.log2(i + 2) for i, item in enumerate(items) if item in actual_set)

def ndcg_at_k(recommended: List[Any], actual: List[Any], k: int) -> float:
    if k <= 0 or not actual: return 0.0
    actual_set = set(actual)
    dcg = _calculate_dcg(recommended[:k], actual_set)
    # IDCG is DCG of sorted relevant items
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(recommended: List[Any], actual: List[Any], k: int) -> float:
    if k <= 0: return 0.0
    actual_set = set(actual)
    for item in recommended[:k]:
        if item in actual_set: return 1.0
    return 0.0

def reciprocal_rank(recommended: List[Any], actual: List[Any]) -> float:
    actual_set = set(actual)
    for i, item in enumerate(recommended):
        if item in actual_set: return 1.0 / (i + 1)
    return 0.0

def catalogue_coverage(recommendations: Dict[Any, List[Any]], catalogue: List[Any]) -> float:
    if not catalogue: return 0.0
    recommended_items = {item for recs in recommendations.values() for item in recs}
    covered_items = recommended_items.intersection(set(catalogue))
    return len(covered_items) / len(catalogue)

def _evaluate_single_user(recs: List[Any], gt: List[Any], k: int, user_id: str) -> Dict[str, Any]:
    return {
        "user_id": str(user_id),
        "precision": float(precision_at_k(recs, gt, k)),
        "recall": float(recall_at_k(recs, gt, k)),
        "f1": float(f1_at_k(recs, gt, k)),
        "ndcg": float(ndcg_at_k(recs, gt, k)),
        "ap": float(average_precision(recs, gt)),
        "hit_rate": float(hit_rate_at_k(recs, gt, k)),
        "rr": float(reciprocal_rank(recs, gt)),
    }

def evaluate_recommendations(
    recommendations: Dict[Any, List[Any]], 
    ground_truth: Dict[Any, List[Any]], 
    k: int = 10
) -> pa.Table:
    all_users = sorted(list(set(recommendations.keys()).union(ground_truth.keys())))
    if not all_users:
        return pa.Table.from_batches([], schema=pa.schema([
            ("user_id", pa.string()), ("precision", pa.float64()), ("recall", pa.float64()),
            ("f1", pa.float64()), ("ndcg", pa.float64()), ("ap", pa.float64()),
            ("hit_rate", pa.float64()), ("rr", pa.float64()),
        ]))

    results = [_evaluate_single_user(recommendations.get(u, []), ground_truth.get(u, []), k, u) for u in all_users]
    
    # Calculate average row
    numeric_cols = ["precision", "recall", "f1", "ndcg", "ap", "hit_rate", "rr"]
    avg_row = {"user_id": "AVERAGE"}
    for col in numeric_cols:
        avg_row[col] = float(np.mean([r[col] for r in results]))
    results.append(avg_row)
        
    return pa.Table.from_pylist(results)
