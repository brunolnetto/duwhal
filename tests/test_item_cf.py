"""Tests for item-based collaborative filtering."""

import pytest
import pandas as pd
import pyarrow as pa

from duwhal.recommenders.item_cf import ItemCF


class TestItemCF:

    @pytest.mark.parametrize("metric", ["jaccard", "cosine", "lift"])
    def test_fit_does_not_raise(self, loaded_conn, metric):
        cf = ItemCF(loaded_conn, metric=metric, min_cooccurrence=1)
        cf.fit()  # should not raise

    def test_recommend_returns_dataframe(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=3)
        assert isinstance(result, pa.Table)

    def test_recommend_columns(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=3)
        assert set(result.column_names) >= {"item_id", "score"}

    def test_recommend_excludes_seed_by_default(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=10)
        item_ids = result.column("item_id").to_pylist() if result.num_rows > 0 else []
        assert "milk" not in item_ids

    def test_recommend_includes_seed_when_not_excluded(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=10, exclude_seed=False)
        # milk may or may not appear depending on similarity, just check no error
        assert isinstance(result, pa.Table)

    def test_similarity_scores_in_range(self, loaded_conn):
        cf = ItemCF(loaded_conn, metric="jaccard", min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=10)
        if result.num_rows > 0:
            for val in result.column("score"):
                assert 0.0 <= val.as_py() <= 1.0

    def test_cosine_similarity_in_range(self, loaded_conn):
        cf = ItemCF(loaded_conn, metric="cosine", min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=10)
        if result.num_rows > 0:
            for val in result.column("score"):
                # Cosine should be roughly 0..1 for positive counts, might be slightly >1 due to precision? No.
                assert 0.0 <= val.as_py() <= 1.0000001

    def test_n_limits_results(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk"], n=2)
        assert result.num_rows <= 2

    def test_empty_seed_returns_empty(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend([], n=5)
        assert result.num_rows == 0

    def test_get_similar_items(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.get_similar_items("milk", n=3)
        assert isinstance(result, pa.Table)
        assert set(result.column_names) >= {"item_id", "score"}
        assert result.num_rows <= 3

    def test_recommend_without_fit_auto_fits(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        # Not calling fit() explicitly
        result = cf.recommend(["milk"], n=3)
        assert isinstance(result, pa.Table)

    def test_invalid_metric_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            ItemCF(loaded_conn, metric="euclidean")

    def test_multiple_seed_items(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        cf.fit()
        result = cf.recommend(["milk", "bread"], n=5)
        if result.num_rows > 0:
            item_ids = result.column("item_id").to_pylist()
            assert "milk" not in item_ids
            assert "bread" not in item_ids

    def test_beer_diapers_similarity(self, loaded_conn):
        # beer and diapers co-occur 3 times — should be similar
        cf = ItemCF(loaded_conn, metric="jaccard", min_cooccurrence=1)
        cf.fit()
        similar = cf.get_similar_items("beer", n=10)
        item_ids = similar.column("item_id").to_pylist() if similar.num_rows > 0 else []
        assert "diapers" in item_ids

    def test_top_k_similar_limits_storage(self, loaded_conn):
        cf = ItemCF(loaded_conn, min_cooccurrence=1, top_k_similar=2)
        cf.fit()
        # Each item should have at most 2 pre-computed neighbors
        # query still returns Arrow table in tests too (assuming Test runs with patched connection?)
        # Actually loaded_conn is DuckDBConnection, query returns Table now.
        count_per_item = loaded_conn.query(
            "SELECT item_a, COUNT(*) as cnt FROM _item_similarity GROUP BY item_a"
        )
        for val in count_per_item.column("cnt"):
            assert val.as_py() <= 2

    def test_item_cf_graph_optimization(self, loaded_conn):
        """Test ItemCF using graph optimization side-channel."""
        from duwhal.recommenders.graph import GraphRecommender
        # Build graph first to create _item_adjacency
        gr = GraphRecommender(loaded_conn)
        gr.build()
        
        cf = ItemCF(loaded_conn, min_cooccurrence=1)
        # This should hit the optimized _build_cooccurrence because _item_adjacency exists
        cf.fit()
        assert cf._fitted
