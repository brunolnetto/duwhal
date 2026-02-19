"""Tests for frequent itemset mining."""

import pytest
import pandas as pd
import pyarrow as pa

from duwhal.mining.frequent_itemsets import FrequentItemsets


class TestFrequentItemsets:

    def test_returns_dataframe(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.3)
        result = fi.fit()
        assert isinstance(result, pa.Table)
        assert set(result.column_names) >= {"itemset", "support", "length"}

    def test_1itemsets_correct_support(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.1)
        result = fi.fit()
        rows = result.to_pylist()
        one_items = [r for r in rows if r["length"] == 1]

        # milk appears in 6/10 transactions → support = 0.6
        milk_row = next((r for r in one_items if r["itemset"] == "milk"), None)
        assert milk_row is not None
        assert abs(milk_row["support"] - 0.6) < 1e-9

        # bread appears in 7/10 → support = 0.7
        bread_row = next((r for r in one_items if r["itemset"] == "bread"), None)
        assert bread_row is not None
        assert abs(bread_row["support"] - 0.7) < 1e-9

    def test_min_support_filters_correctly(self, loaded_conn):
        # beer appears in 3/10 = 0.3 → should be included at threshold 0.25
        fi = FrequentItemsets(loaded_conn, min_support=0.25)
        result = fi.fit()
        rows = result.to_pylist()
        one_items_set = {r["itemset"] for r in rows if r["length"] == 1}
        assert "beer" in one_items_set

        # At 0.35 threshold, beer should be excluded
        fi2 = FrequentItemsets(loaded_conn, min_support=0.35)
        result2 = fi2.fit()
        rows2 = result2.to_pylist()
        one_items_set2 = {r["itemset"] for r in rows2 if r["length"] == 1}
        assert "beer" not in one_items_set2

    def test_2itemsets_generated(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.2)
        result = fi.fit()
        # count length=2
        len_col = result.column("length").to_pylist()
        assert 2 in len_col

    def test_milk_bread_pair_support(self, loaded_conn):
        # milk+bread appear together in T1,T2,T6,T9 → 4/10 = 0.4
        fi = FrequentItemsets(loaded_conn, min_support=0.1)
        result = fi.fit()
        rows = result.to_pylist()
        
        # itemset stored sorted pipe-delimited
        match = next((r for r in rows if r["itemset"] in ["bread|milk", "milk|bread"] and r["length"] == 2), None)
        assert match is not None
        assert abs(match["support"] - 0.4) < 1e-9

    def test_max_len_limits_output(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.1, max_len=1)
        result = fi.fit()
        # Check max value in length column
        import pyarrow.compute as pc
        max_len = pc.max(result.column("length")).as_py()
        assert max_len == 1

    def test_max_len_2(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.1, max_len=2)
        result = fi.fit()
        import pyarrow.compute as pc
        max_len = pc.max(result.column("length")).as_py()
        assert max_len <= 2

    def test_support_always_between_0_and_1(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.05)
        result = fi.fit()
        for val in result.column("support"):
            assert 0 < val.as_py() <= 1

    def test_invalid_min_support_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            FrequentItemsets(loaded_conn, min_support=0)
        with pytest.raises(ValueError):
            FrequentItemsets(loaded_conn, min_support=1.5)

    def test_high_min_support_empty_result(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.99)
        result = fi.fit()
        assert result.num_rows == 0

    def test_last_sql_populated(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.3)
        fi.fit()
        assert len(fi.last_sql_) > 0
        assert all(isinstance(s, str) for s in fi.last_sql_)

    def test_length_column_correct(self, loaded_conn):
        fi = FrequentItemsets(loaded_conn, min_support=0.1)
        result = fi.fit()
        rows = result.to_pylist()
        for row in rows:
            expected_len = len(row["itemset"].split("|"))
            assert row["length"] == expected_len

    def test_frequent_itemsets_graph_optimization(self, loaded_conn):
        """Test FrequentItemsets using graph optimization side-channel."""
        from duwhal.recommenders.graph import GraphRecommender
        # Build graph first to create _item_adjacency
        gr = GraphRecommender(loaded_conn)
        gr.build()
        
        fi = FrequentItemsets(loaded_conn, min_support=0.1)
        # This should hit the optimized _create_freq2 because _item_adjacency exists
        result = fi.fit()
        assert len(result) > 0
