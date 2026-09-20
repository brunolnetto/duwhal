"""Performance regression suite.

These tests guard against accidental performance degradation by asserting that
key operations complete within generous wall-clock budgets on the test runner.
"""

import time

import pytest

from duwhal import Duwhal
from duwhal.datasets import generate_large_scale_data


@pytest.fixture
def large_df():
    return generate_large_scale_data(n_transactions=10_000, n_items=500, seed=42)


class TestPerformanceRegression:
    """Ensure core operations stay within budget on modest hardware."""

    def test_ingestion_10k_rows(self, large_df):
        with Duwhal() as db:
            start = time.perf_counter()
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 5_000, f"ingestion took {elapsed_ms:.1f} ms"

    def test_fit_cf_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            start = time.perf_counter()
            db.fit_cf(metric="jaccard", min_cooccurrence=2, top_k_similar=20)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 10_000, f"fit_cf took {elapsed_ms:.1f} ms"

    def test_fit_graph_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            start = time.perf_counter()
            db.fit_graph(min_cooccurrence=2)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 10_000, f"fit_graph took {elapsed_ms:.1f} ms"

    def test_recommend_graph_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            db.fit_graph(min_cooccurrence=2)
            start = time.perf_counter()
            recs = db.recommend_graph([large_df["item_id"].iloc[0]], n=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 10_000, f"recommend_graph took {elapsed_ms:.1f} ms"
            assert recs.num_rows <= 10

    def test_fit_popularity_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            start = time.perf_counter()
            db.fit_popularity()
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 2_000, f"fit_popularity took {elapsed_ms:.1f} ms"

    def test_batch_recommend_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            db.fit_cf(metric="jaccard", min_cooccurrence=2, top_k_similar=20)
            seeds = [[item] for item in large_df["item_id"].unique()[:5]]
            start = time.perf_counter()
            results = db.recommend_batch(seeds, strategy="cf", n=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 2_000, f"recommend_batch took {elapsed_ms:.1f} ms"
            assert len(results) == len(seeds)
