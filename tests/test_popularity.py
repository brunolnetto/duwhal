"""Tests for popularity-based recommender."""

import pytest
import pandas as pd
import pyarrow as pa
from datetime import datetime, timedelta

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders.popularity import PopularityRecommender


@pytest.fixture
def timestamped_conn():
    """Connection with timestamped transactions."""
    conn = DuckDBConnection()
    base_time = datetime(2024, 1, 1)
    rows = []
    # Recent items (last 30 days): item_A x5, item_B x3
    for i in range(5):
        rows.append({"order_id": f"new_{i}", "item_id": "item_A",
                     "event_time": base_time + timedelta(days=350 + i)})
        rows.append({"order_id": f"new_{i}", "item_id": "item_B",
                     "event_time": base_time + timedelta(days=350 + i)})
    # Old item: item_C x10 but old
    for i in range(10):
        rows.append({"order_id": f"old_{i}", "item_id": "item_C",
                     "event_time": base_time + timedelta(days=i)})

    df = pd.DataFrame(rows)
    conn.register("_tmp", df)
    # Manual table creation with correct schema
    conn.execute("""
        CREATE TABLE interactions AS
        SELECT order_id AS set_id, item_id AS node_id, event_time::TIMESTAMP AS event_time FROM _tmp
    """)
    yield conn
    conn.close()


class TestPopularityRecommender:

    def test_global_fit_and_recommend(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=5)
        assert isinstance(result, pa.Table)
        # item_id is aliased from node_id in recommend() return
        assert set(result.column_names) >= {"item_id", "score", "rank"}

    def test_recommend_n_limit(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=3)
        assert result.num_rows <= 3

    def test_scores_between_0_and_1(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=10)
        for val in result.column("score"):
            assert 0.0 <= val.as_py() <= 1.0

    def test_rank_1_is_most_popular(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=10)
        # bread appears in 7/10 orders → most popular
        # filter where rank == 1
        rows = result.to_pylist()
        top_item = next(r["item_id"] for r in rows if r["rank"] == 1)
        assert top_item == "bread"

    def test_exclude_items(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=10, exclude_items=["bread", "milk"])
        item_ids = result.column("item_id").to_pylist() if result.num_rows > 0 else []
        assert "bread" not in item_ids
        assert "milk" not in item_ids

    def test_auto_fit_on_recommend(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        # Not calling fit() explicitly
        result = pop.recommend(n=3)
        assert isinstance(result, pa.Table)

    def test_trending_strategy(self, timestamped_conn):
        # timestamped_conn has interactions table with set_id, node_id, event_time
        pop = PopularityRecommender(
            timestamped_conn,
            strategy="trending",
            timestamp_col="event_time",
            window_days=30,
        )
        pop.fit()
        result = pop.recommend(n=3)
        # item_C is old → should not appear in trending top result
        item_ids = result.column("item_id").to_pylist() if result.num_rows > 0 else []
        if item_ids:
            assert "item_C" != item_ids[0]

    def test_trending_requires_timestamp_col(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn, strategy="trending")
        with pytest.raises(ValueError, match="timestamp_col"):
            pop.fit()

    def test_invalid_strategy_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            PopularityRecommender(loaded_conn, strategy="unknown")

    def test_ranks_are_monotonic(self, loaded_conn):
        pop = PopularityRecommender(loaded_conn)
        pop.fit()
        result = pop.recommend(n=10)
        ranks = result.column("rank").to_pylist()
        assert ranks == sorted(ranks)

    def test_popularity_trending_sort_fallback(self, conn):
        """Test fallback to sort_column for trending strategy."""
        from duwhal.core.ingestion import load_interactions
        df = pd.DataFrame({
            "set_id": ["S1", "S1"],
            "node_id": ["A", "B"],
            "sort_column": [datetime.now(), datetime.now()]
        })
        load_interactions(conn, df, sort_col="sort_column")
        pop = PopularityRecommender(conn, strategy="trending")
        # Should fallback to sort_column because timestamp_col is not provided
        pop.fit()
        assert pop._fitted
