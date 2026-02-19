"""Integration tests for the high-level Duwhal API."""

import pytest
import pandas as pd
import pyarrow as pa
import narwhals as nw

from duwhal.api import Duwhal


class TestDuwhalIngestion:

    def test_load_interactions_returns_count(self, transactions_df):
        with Duwhal() as db:
            count = db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            assert count == len(transactions_df)

    def test_load_matrix(self):
        matrix = pd.DataFrame({
            "milk":  [1, 0, 1],
            "bread": [1, 1, 0],
        }, index=["T1", "T2", "T3"])
        matrix.index.name = "set_id"
        with Duwhal() as db:
            count = db.load_interaction_matrix(matrix)
            assert count == 4  # T1:2 items, T2:1 item, T3:1 item

    def test_raw_sql(self, transactions_df):
        with Duwhal() as db:
            db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            result = db.sql("SELECT COUNT(*) AS n FROM interactions")
            # Arrow table access
            assert result.column("n")[0].as_py() == len(transactions_df)


class TestDuwhalMining:

    def test_frequent_itemsets(self, duwhal_instance):
        result = duwhal_instance.frequent_itemsets(min_support=0.3)
        assert isinstance(result, pa.Table)
        assert result.num_rows > 0

    def test_association_rules(self, duwhal_instance):
        rules = duwhal_instance.association_rules(
            min_support=0.1, min_confidence=0.1, min_lift=0.0
        )
        assert isinstance(rules, pa.Table)

    def test_association_rules_cached(self, duwhal_instance):
        duwhal_instance.association_rules(min_support=0.1, min_confidence=0.1, min_lift=0.0)
        assert duwhal_instance._rules is not None


class TestDuwhalRecommendByRules:

    def test_recommend_by_rules_basic(self, duwhal_instance):
        duwhal_instance.association_rules(
            min_support=0.1, min_confidence=0.1, min_lift=0.0
        )
        result = duwhal_instance.recommend_by_rules(["milk"], n=5)
        assert isinstance(result, pa.Table)

    def test_recommend_by_rules_excludes_seed(self, duwhal_instance):
        duwhal_instance.association_rules(
            min_support=0.1, min_confidence=0.1, min_lift=0.0
        )
        result = duwhal_instance.recommend_by_rules(["milk"], n=5)
        if result.num_rows > 0:
            item_ids = result.column("item_id").to_pylist()
            assert "milk" not in item_ids

    def test_recommend_by_rules_without_rules_raises(self, duwhal_instance):
        with pytest.raises(RuntimeError):
            duwhal_instance.recommend_by_rules(["milk"])

    def test_recommend_by_rules_invalid_metric(self, duwhal_instance):
        duwhal_instance.association_rules(min_support=0.1, min_confidence=0.1, min_lift=0.0)
        with pytest.raises(ValueError):
            duwhal_instance.recommend_by_rules(["milk"], metric="invalid")

    def test_recommend_by_rules_n_limit(self, duwhal_instance):
        duwhal_instance.association_rules(min_support=0.05, min_confidence=0.05, min_lift=0.0)
        result = duwhal_instance.recommend_by_rules(["milk"], n=2)
        assert result.num_rows <= 2

    def test_recommend_by_rules_columns(self, duwhal_instance):
        duwhal_instance.association_rules(min_support=0.1, min_confidence=0.1, min_lift=0.0)
        result = duwhal_instance.recommend_by_rules(["milk"], n=5)
        if result.num_rows > 0:
            assert set(result.column_names) >= {"item_id", "score", "rule"}


class TestDuwhalCF:

    def test_fit_cf_and_recommend(self, duwhal_instance):
        duwhal_instance.fit_cf()
        result = duwhal_instance.recommend_cf(["milk"], n=3)
        assert isinstance(result, pa.Table)

    def test_recommend_cf_without_fit_raises(self, duwhal_instance):
        with pytest.raises(RuntimeError):
            duwhal_instance.recommend_cf(["milk"])

    def test_fit_cf_returns_self(self, duwhal_instance):
        result = duwhal_instance.fit_cf()
        assert result is duwhal_instance

    def test_cf_metric_options(self, duwhal_instance):
        for metric in ["jaccard", "cosine", "lift"]:
            db = Duwhal()
            # Still use pandas for input if available in test env
            db.load_interactions(pd.DataFrame([
                ("T1", "a"), ("T1", "b"), ("T2", "a"), ("T2", "b"), ("T3", "b"), ("T3", "c")
            ], columns=["order_id", "item_id"]), set_col="order_id", node_col="item_id")
            db.fit_cf(metric=metric)
            result = db.recommend_cf(["a"], n=3)
            assert isinstance(result, pa.Table)
            db.close()


class TestDuwhalPopularity:

    def test_fit_popularity_and_recommend(self, duwhal_instance):
        duwhal_instance.fit_popularity()
        result = duwhal_instance.recommend_popular(n=3)
        assert isinstance(result, pa.Table)

    def test_recommend_popular_without_fit_auto_fits(self, duwhal_instance):
        result = duwhal_instance.recommend_popular(n=3)
        assert isinstance(result, pa.Table)

    def test_fit_popularity_returns_self(self, duwhal_instance):
        result = duwhal_instance.fit_popularity()
        assert result is duwhal_instance


class TestDuwhalContextManager:

    def test_context_manager(self, transactions_df):
        with Duwhal() as db:
            count = db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            assert count > 0

    def test_repr(self):
        db = Duwhal()
        assert "Duwhal" in repr(db)
        db.close()


class TestCoverageGaps:

    def test_item_cf_get_similar_auto_fit(self, duwhal_instance):
        """Test get_similar_items calls fit() if needed."""
        # We need a fresh instance that hasn't been fitted
        from duwhal.recommenders.item_cf import ItemCF
        cf = ItemCF(duwhal_instance.conn, min_cooccurrence=1)
        res = cf.get_similar_items("milk", n=1)
        assert isinstance(res, pa.Table)
        assert cf._fitted

    def test_item_cf_unknown_metric_execution(self, duwhal_instance):
        """Test _metric_sql raises ValueError for unknown metric (bypass init check)."""
        from duwhal.recommenders.item_cf import ItemCF
        cf = ItemCF(duwhal_instance.conn)
        cf.metric = "unknown_metric"  # Bypass __init__ check
        with pytest.raises(ValueError, match="Unknown metric"):
            cf.fit()

    def test_load_matrix_append(self, conn):
        """Test load_interaction_matrix with append=True."""
        df = pd.DataFrame({"set_id": ["T1"], "item_A": [1]})
        from duwhal.core.ingestion import load_interaction_matrix
        load_interaction_matrix(conn, df, table_name="basket_append")
        count = load_interaction_matrix(conn, df, table_name="basket_append", append=True)
        assert count == 2

    def test_load_matrix_missing_set_id(self, conn):
        """Test load_interaction_matrix behaves when set_id is missing (coverage for pass block)."""
        import polars as pl
        # Polars DF without set_id -> bypasses pandas index logic
        df = pl.DataFrame({"item_A": [1]})
        from duwhal.core.ingestion import load_interaction_matrix
        # This will fail when Narwhals tries to unpivot with missing index col
        with pytest.raises(ValueError, match="Input DataFrame must have a 'set_id' column or index"):
             load_interaction_matrix(conn, df)

    def test_collect_all_levels_empty(self, duwhal_instance):
        """Test _collect_all_levels directly to cover empty parts path."""
        from duwhal.mining.frequent_itemsets import FrequentItemsets
        fi = FrequentItemsets(duwhal_instance.conn)
        # Calling without fit() -> no tables exist -> parts empty
        res = fi._collect_all_levels(max_level=1)
        assert res.num_rows == 0
        assert "itemset" in res.column_names

    def test_frequent_itemsets_empty(self, duwhal_instance):
        """Test frequent_itemsets returning empty DF."""
        # High support -> no frequent items
        res = duwhal_instance.frequent_itemsets(min_support=1.0)
        assert res.num_rows == 0
        assert "itemset" in res.column_names

    def test_sequences_max_gap(self):
        """Test sequential_patterns with max_gap."""
        df = pd.DataFrame([
            ("T1", "A", "2024-01-01 10:00"),
            ("T1", "B", "2024-01-01 10:05"), # Gap 1
            ("T1", "C", "2024-01-01 10:10"), # Gap A->C is > 1?
        ], columns=["order_id", "item_id", "ts"])
        
        from duwhal.api import Duwhal
        with Duwhal() as db:
             db.conn.register("source", df)
             # Manually create table with new schema
             db.conn.execute("CREATE TABLE interactions AS SELECT order_id AS set_id, item_id AS node_id, ts::TIMESTAMP AS ts FROM source")
             # max_gap=0
             res = db.sequential_patterns(timestamp_col="ts", min_support=0.01, max_gap=0)
             
             # Convert to dict for easier checking
             res_pylist = res.to_pylist()
             pairs = set((r["prefix"], r["suffix"]) for r in res_pylist)
             
             assert ("A", "B") in pairs
             assert ("A", "C") not in pairs


class TestDuwhalEndToEnd:

    def test_full_pipeline_rules(self, transactions_df):
        """Full pipeline: load → mine rules → recommend."""
        with Duwhal() as db:
            db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            rules = db.association_rules(
                min_support=0.1, min_confidence=0.1, min_lift=0.0
            )
            assert rules.num_rows > 0
            recs = db.recommend_by_rules(["milk"], n=5)
            assert isinstance(recs, pa.Table)

    def test_full_pipeline_cf(self, transactions_df):
        """Full pipeline: load → fit CF → recommend."""
        with Duwhal() as db:
            db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            db.fit_cf(metric="jaccard")
            recs = db.recommend_cf(["milk", "bread"], n=5)
            assert isinstance(recs, pa.Table)

    def test_persistent_database_pipeline(self, transactions_df, tmp_path):
        """End-to-end with a persistent DuckDB file."""
        db_path = str(tmp_path / "store.duckdb")
        with Duwhal(database=db_path) as db:
            db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
            rules = db.association_rules(min_support=0.1, min_confidence=0.1, min_lift=0.0)
            assert isinstance(rules, pa.Table)

class TestDuwhalSequences:

    def test_sequential_patterns(self):
        df = pd.DataFrame([
            ("T1", "A", "2024-01-01 10:00"),
            ("T1", "B", "2024-01-01 10:05"),
            ("T2", "A", "2024-01-01 11:00"),
            ("T2", "B", "2024-01-01 11:05"),
        ], columns=["order_id", "item_id", "ts"])
        
        with Duwhal() as db:
             db.conn.register("source", df)
             # Manual table with proper schema
             db.conn.execute("CREATE TABLE interactions AS SELECT order_id AS set_id, item_id AS node_id, ts::TIMESTAMP AS ts FROM source")
             res = db.sequential_patterns(timestamp_col="ts", min_support=0.1)
             assert res.num_rows > 0
             prefixes = res.column("prefix").to_pylist()
             suffixes = res.column("suffix").to_pylist()
             assert "A" in prefixes
             assert "B" in suffixes

    def test_recommend_by_rules_empty_matches(self, duwhal_instance):
         duwhal_instance.association_rules(min_support=0.01)
         # Recommend for item not in rules
         res = duwhal_instance.recommend_by_rules(["nonexistent_item"])
         assert res.num_rows == 0
         assert "item_id" in res.column_names

class TestAPICoverage:
    def test_api_recommend_by_rules_break(self, duwhal_instance):
        """Test that the loop breaks when n limits are reached."""
        # Create enough rules to exceed n=1
        duwhal_instance.load_interactions(pd.DataFrame({
            "order_id": ["T1", "T1", "T1", "T2", "T2", "T2"],
            "item_id":  ["A", "B", "C", "A", "B", "C"]
        }), set_col="order_id", node_col="item_id")
        # A, B, C are fully correlated.
        # Rules: A->B, A->C, B->A, B->C, C->A, C->B
        duwhal_instance.association_rules(min_support=0.01, min_confidence=0.01)
        
        # Recommendations for [A]: should get B and C.
        # Set n=1. Loop should break after finding 1.
        res = duwhal_instance.recommend_by_rules(["A"], n=1)
        assert res.num_rows == 1
