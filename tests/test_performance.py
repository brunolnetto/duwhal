"""Performance regression suite.

These tests guard against accidental performance degradation by asserting that
key operations complete within generous wall-clock budgets on the test runner.
"""

import time

import numpy as np
import pandas as pd
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
            assert elapsed_ms < 1_000, f"recommend_graph took {elapsed_ms:.1f} ms"
            assert recs.num_rows <= 10

    def test_recommend_cf_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            db.fit_cf(metric="jaccard", min_cooccurrence=2, top_k_similar=20)
            start = time.perf_counter()
            recs = db.recommend([large_df["item_id"].iloc[0]], strategy="cf", n=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 500, f"recommend_cf took {elapsed_ms:.1f} ms"
            assert recs.num_rows <= 10

    def test_recommend_popularity_10k_rows(self, large_df):
        with Duwhal() as db:
            db.load_interactions(large_df, set_col="order_id", node_col="item_id")
            db.fit_popularity()
            start = time.perf_counter()
            recs = db.recommend(strategy="popularity", n=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, f"popularity recommend took {elapsed_ms:.1f} ms"
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
            assert results.num_rows == len(seeds) * 5


def _percentile(values, p):
    return float(np.percentile(values, p))


def _benchmark(db, fn, seeds, n=10, repeats=20):
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(seeds, n=n)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
    }


def _synthetic_kuairec_like(seed=42):
    rng = np.random.default_rng(seed)
    n_users = 100
    n_items = 1_000
    max_session = 15
    interactions = []
    for user_id in range(n_users):
        n_interactions = rng.integers(400, 800)
        session_id = f"u{user_id}"
        items = rng.integers(0, n_items, size=n_interactions)
        # Break into sessions of at most max_session items.
        for i, item in enumerate(items):
            if i % max_session == 0:
                session_id = f"u{user_id}_s{i // max_session}"
            interactions.append({"user_id": session_id, "item_id": f"i{item}"})
    return pd.DataFrame(interactions)


class TestLatencyDistributionBenchmarks:
    """Record p50/p95/p99 latencies for serving paths (best-effort assertions)."""

    def test_itemcf_latency_distribution(self):
        df = _synthetic_kuairec_like(seed=1)
        with Duwhal() as db:
            db.load_interactions(df, set_col="user_id", node_col="item_id")
            db.fit_cf(metric="jaccard", min_cooccurrence=2, top_k_similar=20)
            stats = _benchmark(db, db.recommend, ["i0"], n=10, repeats=10)
            assert stats["p50"] < 500, f"ItemCF p50={stats['p50']:.1f} ms"

    def test_graph_bounded_depth1_latency_distribution(self):
        df = _synthetic_kuairec_like(seed=2)
        with Duwhal() as db:
            db.load_interactions(df, set_col="user_id", node_col="item_id")
            db.fit_graph(min_cooccurrence=2, top_k_edges=50)
            stats = _benchmark(db, db.recommend_graph, ["i0"], n=10, repeats=10)
            assert stats["p95"] < 1_000, f"Graph depth=1 p95={stats['p95']:.1f} ms"

    def test_graph_bounded_depth2_beam50_latency_distribution(self):
        df = _synthetic_kuairec_like(seed=3)
        with Duwhal() as db:
            db.load_interactions(df, set_col="user_id", node_col="item_id")
            db.fit_graph(min_cooccurrence=2, top_k_edges=50)

            def recommend_fn(seeds, n):
                return db.recommend_graph(seeds, n=n, max_depth=2, beam_width=50)

            stats = _benchmark(db, recommend_fn, ["i0"], n=10, repeats=10)
            assert stats["p95"] < 1_000, f"Graph depth=2 beam=50 p95={stats['p95']:.1f} ms"


class TestPathologicalGraphBenchmarks:
    """Ensure bounded graph does not explode on high-degree hubs."""

    def test_hub_graph_recommendation_bounded(self):
        rng = np.random.default_rng(42)
        n_hubs = 5
        n_per_hub = 500
        n_contexts = 5_000
        edges = []
        for ctx in range(n_contexts):
            # Each context contains one hub and a small random tail.
            hub = f"hub{ctx % n_hubs}"
            tail = rng.choice([f"tail{i}" for i in range(n_per_hub)], size=10, replace=False)
            edges.append({"order_id": f"c{ctx}", "item_id": hub})
            for item in tail:
                edges.append({"order_id": f"c{ctx}", "item_id": item})
        df = pd.DataFrame(edges)
        with Duwhal() as db:
            db.load_interactions(df, set_col="order_id", node_col="item_id")
            db.fit_graph(min_cooccurrence=2, top_k_edges=100)
            start = time.perf_counter()
            recs = db.recommend_graph(["hub0"], n=10, max_depth=4, beam_width=200)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 1_000, f"hub graph recommend took {elapsed_ms:.1f} ms"
            assert recs.num_rows <= 10


class TestKuaiRecWorkloadBenchmark:
    """Approximate the KuaiRec small-user workload (train ~60k interactions)."""

    def test_kuairec_itemcf_and_graph_latency(self):
        rng = np.random.default_rng(7)
        n_users = 100
        n_items = 2_000
        target_interactions = 60_000
        interactions = []
        for user_id in range(n_users):
            n = target_interactions // n_users
            popular_items = rng.choice(n_items, size=n, replace=True, p=np.linspace(2, 1, n_items) / np.linspace(2, 1, n_items).sum())
            for item in popular_items:
                interactions.append({"user_id": f"u{user_id}", "item_id": f"i{item}"})
        df = pd.DataFrame(interactions)
        with Duwhal() as db:
            db.load_interactions(df, set_col="user_id", node_col="item_id")
            db.fit_cf(metric="jaccard", min_cooccurrence=2, top_k_similar=20)
            seeds = [f"i{rng.integers(0, n_items)}" for _ in range(5)]
            cf_times = []
            for _ in range(20):
                start = time.perf_counter()
                db.recommend(seeds, strategy="cf", n=10)
                cf_times.append((time.perf_counter() - start) * 1000)
            cf_p50 = _percentile(cf_times, 50)
            assert cf_p50 < 200, f"KuaiRec CF p50={cf_p50:.1f} ms"

            db.fit_graph(min_cooccurrence=2, top_k_edges=100)
            graph_times = []
            for _ in range(20):
                start = time.perf_counter()
                db.recommend_graph(seeds, n=10, max_depth=2, beam_width=200)
                graph_times.append((time.perf_counter() - start) * 1000)
            graph_p95 = _percentile(graph_times, 95)
            assert graph_p95 < 1_000, f"KuaiRec Graph p95={graph_p95:.1f} ms"
