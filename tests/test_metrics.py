"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
import pyarrow as pa

from duwhal.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    ndcg_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    evaluate_recommendations,
    catalogue_coverage,
)


class TestPrecisionAtK:

    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_precision(self):
        assert abs(precision_at_k(["a", "x"], ["a", "b"], k=2) - 0.5) < 1e-9

    def test_k_truncates(self):
        # only top-1 is relevant
        assert precision_at_k(["a", "b"], ["a"], k=1) == 1.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_more_relevant_than_k(self):
        # k=2, 2 relevant hits out of 2 → precision = 1.0
        assert precision_at_k(["a", "b", "c"], ["a", "b", "d"], k=2) == 1.0


class TestRecallAtK:

    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x"], ["a", "b"], k=1) == 0.0

    def test_partial_recall(self):
        assert abs(recall_at_k(["a", "x"], ["a", "b"], k=2) - 0.5) < 1e-9

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0


class TestF1AtK:

    def test_perfect_f1(self):
        assert f1_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_zero_f1(self):
        assert f1_at_k(["x"], ["a"], k=1) == 0.0

    def test_f1_harmonic_mean(self):
        p = precision_at_k(["a", "x"], ["a", "b"], k=2)
        r = recall_at_k(["a", "x"], ["a", "b"], k=2)
        expected = 2 * p * r / (p + r)
        assert abs(f1_at_k(["a", "x"], ["a", "b"], k=2) - expected) < 1e-9


class TestAveragePrecision:

    def test_perfect_ap(self):
        assert average_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_ap(self):
        assert average_precision(["x", "y"], ["a", "b"]) == 0.0

    def test_ap_order_matters(self):
        ap1 = average_precision(["a", "b", "x"], ["a", "b"])
        ap2 = average_precision(["x", "a", "b"], ["a", "b"])
        assert ap1 > ap2  # hitting earlier is better

    def test_empty_relevant(self):
        assert average_precision(["a"], []) == 0.0


class TestNDCGAtK:

    def test_perfect_ndcg(self):
        assert abs(ndcg_at_k(["a", "b"], ["a", "b"], k=2) - 1.0) < 1e-9

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_ndcg_decreasing_with_worse_ranking(self):
        ndcg_good = ndcg_at_k(["a", "x"], ["a"], k=2)
        ndcg_bad = ndcg_at_k(["x", "a"], ["a"], k=2)
        assert ndcg_good > ndcg_bad

    def test_ndcg_between_0_and_1(self):
        score = ndcg_at_k(["a", "b", "c"], ["a", "z"], k=3)
        assert 0 <= score <= 1


class TestHitRateAtK:

    def test_hit(self):
        assert hit_rate_at_k(["a", "b"], ["a"], k=2) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_first_item_hit(self):
        assert hit_rate_at_k(["a"], ["a"], k=1) == 1.0


class TestReciprocalRank:

    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5

    def test_third_position(self):
        assert abs(reciprocal_rank(["x", "y", "a"], ["a"]) - 1/3) < 1e-9

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestEvaluateRecommendations:

    @pytest.fixture
    def sample_eval_data(self):
        recommendations = {
            "user1": ["a", "b", "c", "d"],
            "user2": ["x", "a", "b", "c"],
            "user3": ["a", "b", "c", "d"],
        }
        ground_truth = {
            "user1": ["a", "c"],
            "user2": ["a", "b"],
            "user3": ["z"],  # no hits
        }
        return recommendations, ground_truth

    def test_returns_dataframe(self, sample_eval_data):
        recs, gt = sample_eval_data
        result = evaluate_recommendations(recs, gt, k=4)
        assert isinstance(result, pa.Table)

    def test_has_average_row(self, sample_eval_data):
        recs, gt = sample_eval_data
        result = evaluate_recommendations(recs, gt, k=4)
        user_ids = result.column("user_id").to_pylist()
        assert "AVERAGE" in user_ids

    def test_metric_columns_present(self, sample_eval_data):
        recs, gt = sample_eval_data
        result = evaluate_recommendations(recs, gt, k=4)
        for col in ["precision", "recall", "f1", "ndcg", "ap", "hit_rate", "rr"]:
            assert col in result.column_names

    def test_all_zeros_when_no_hits(self, sample_eval_data):
        recs, gt = sample_eval_data
        result = evaluate_recommendations(recs, gt, k=4)
        # Filter for user3
        # Use simple python iteration since table is small
        rows = result.to_pylist()
        user3 = next(r for r in rows if r["user_id"] == "user3")
        assert user3["precision"] == 0.0
        assert user3["hit_rate"] == 0.0

    def test_perfect_scores_for_perfect_recs(self):
        recs = {"u1": ["a", "b"]}
        gt = {"u1": ["a", "b"]}
        result = evaluate_recommendations(recs, gt, k=2)
        rows = result.to_pylist()
        u1 = next(r for r in rows if r["user_id"] == "u1")
        assert u1["precision"] == 1.0
        assert u1["recall"] == 1.0

    def test_empty_recommendations(self):
        result = evaluate_recommendations({}, {}, k=5)
        assert result.num_rows == 0


class TestCatalogueCoverage:

    def test_full_coverage(self):
        recs = {"u1": ["a", "b"], "u2": ["c"]}
        catalogue = ["a", "b", "c"]
        assert catalogue_coverage(recs, catalogue) == 1.0

    def test_partial_coverage(self):
        recs = {"u1": ["a"]}
        catalogue = ["a", "b"]
        assert catalogue_coverage(recs, catalogue) == 0.5

    def test_empty_catalogue(self):
        assert catalogue_coverage({"u1": ["a"]}, []) == 0.0

    def test_zero_coverage(self):
        recs = {"u1": ["x"]}
        catalogue = ["a", "b"]
        assert catalogue_coverage(recs, catalogue) == 0.0
