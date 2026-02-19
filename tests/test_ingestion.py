"""Tests for data ingestion."""

import pytest
import pandas as pd
import numpy as np

from duwhal.core.connection import DuckDBConnection
from duwhal.core.ingestion import load_interactions, load_interaction_matrix


class TestLoadInteractions:

    def test_basic_load(self, conn, transactions_df):
        # transactions_df has order_id, item_id
        count = load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id"
        )
        assert count == len(transactions_df)

    def test_table_created(self, conn, transactions_df):
        load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id"
        )
        assert conn.table_exists("interactions")

    def test_column_names_normalised(self, conn):
        df = pd.DataFrame({
            "basket_id": ["A", "A", "B"],
            "product":   ["x", "y", "x"],
        })
        load_interactions(
            conn, df,
            set_col="basket_id", node_col="product"
        )
        cols = conn.query("SELECT * FROM interactions LIMIT 0").column_names
        assert "set_id" in cols
        assert "node_id" in cols

    def test_custom_table_name(self, conn, transactions_df):
        load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id",
            table_name="my_orders"
        )
        assert conn.table_exists("my_orders")

    def test_append_mode(self, conn, transactions_df):
        load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id"
        )
        count2 = load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id",
            append=True
        )
        assert count2 == 2 * len(transactions_df)

    def test_replace_mode(self, conn, transactions_df):
        load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id"
        )
        count2 = load_interactions(
            conn, transactions_df,
            set_col="order_id", node_col="item_id",
            append=False
        )
        assert count2 == len(transactions_df)

    def test_missing_column_raises(self, conn):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing columns"):
            load_interactions(conn, df)

    def test_null_rows_dropped(self, conn):
        df = pd.DataFrame({
            "order_id": ["T1", "T1", None, "T2"],
            "item_id":  ["x",  None,  "y",  "x"],
        })
        # Use simple NaN check
        import numpy as np
        df.replace({None: np.nan}, inplace=True)
        
        count = load_interactions(
            conn, df, 
            set_col="order_id", node_col="item_id"
        )
        # Only rows where both columns are non-null
        assert count == 2

    def test_from_csv(self, conn, transactions_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        transactions_df.to_csv(csv_path, index=False)
        count = load_interactions(
            conn, csv_path,
            set_col="order_id", node_col="item_id"
        )
        assert count == len(transactions_df)

    def test_from_parquet(self, conn, transactions_df, tmp_path):
        pq_path = tmp_path / "data.parquet"
        transactions_df.to_parquet(pq_path, index=False)
        count = load_interactions(
            conn, pq_path,
            set_col="order_id", node_col="item_id"
        )
        assert count == len(transactions_df)


    def test_unsupported_file_type_raises(self, conn, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_interactions(conn, f)

    def test_from_polars(self, conn, transactions_polars):
        count = load_interactions(
            conn, transactions_polars,
            set_col="order_id", node_col="item_id"
        )
        assert count == len(transactions_polars)

    def test_from_polars_lazy(self, conn, transactions_lazy):
        count = load_interactions(
            conn, transactions_lazy,
            set_col="order_id", node_col="item_id"
        )
        # lazy df length is not directly accessible, but we know the data
        assert count == 19

    def test_sort_column_ingestion(self, conn):
        df = pd.DataFrame({
            "order_id": ["O1", "O1"],
            "item_id": ["A", "B"],
            "ts": ["2024-01-01", "2024-01-02"]
        })
        load_interactions(
            conn, df,
            set_col="order_id", node_col="item_id",
            sort_col="ts"
        )
        # Check if sort_column exists and has values
        res = conn.execute("SELECT sort_column FROM interactions ORDER BY sort_column").fetchall()
        assert res[0][0] == "2024-01-01"
        assert res[1][0] == "2024-01-02"

    def test_sort_callback(self, conn):
        df = pd.DataFrame({
            "order_id": ["O1"],
            "item_id": ["A"],
            "ts": ["2024-01-01"]
        })
        # Callback to convert string to something else
        def parser(x):
            return x + "_processed"
            
        load_interactions(
            conn, df,
            set_col="order_id", node_col="item_id",
            sort_col="ts",
            sort_callback=parser
        )
        res = conn.execute("SELECT sort_column FROM interactions").fetchone()[0]
        assert res == "2024-01-01_processed"

    def test_append_with_schema_evolution(self, conn):
        # 1. Load without sort
        df1 = pd.DataFrame({"set": ["S1"], "item": ["A"]})
        load_interactions(conn, df1, set_col="set", node_col="item")
        
        # 2. Append WITH sort
        df2 = pd.DataFrame({"set": ["S2"], "item": ["B"], "ts": [100]})
        load_interactions(conn, df2, set_col="set", node_col="item", sort_col="ts", append=True)
        
        # Check
        res = conn.execute("SELECT set_id, node_id, sort_column FROM interactions ORDER BY set_id").fetchall()
        # S1, A, NULL
        # S2, B, 100
        assert len(res) == 2
        # Use almost equal for float/int flexibility or exact check
        assert res[0][0] == "S1" and res[0][2] is None
        assert res[1][0] == "S2" and res[1][2] == 100



class TestLoadInteractionMatrix:

    def test_basic_basket(self, conn):
        df = pd.DataFrame({
            "milk":   [1, 0, 1],
            "bread":  [1, 1, 0],
            "butter": [0, 1, 0],
        }, index=["T1", "T2", "T3"])
        df.index.name = "set_id" # Help the loader find the set ID
        
        count = load_interaction_matrix(conn, df)
        assert count == 5  # T1:milk+bread, T2:bread+butter, T3:milk

    def test_basket_with_set_id_column(self, conn):
        df = pd.DataFrame({
            "set_id": ["T1", "T2"],
            "milk":     [1, 0],
            "bread":    [1, 1],
        })
        count = load_interaction_matrix(conn, df)
        assert count == 3  # T1:milk+bread, T2:bread

    def test_basket_schema(self, conn):
        df = pd.DataFrame({"item_a": [1], "item_b": [1]}, index=["T1"])
        df.index.name = "set_id"
        load_interaction_matrix(conn, df)
        cols = conn.query("SELECT * FROM interactions LIMIT 0").column_names
        assert "set_id" in cols
        assert "node_id" in cols



from unittest.mock import MagicMock, patch
import narwhals as nw

class TestIngestionCoverage:
    def test_ingestion_load_interactions_fallback_columns(self, conn):
        """Test load_interactions when collect_schema fails, falling back to .columns."""
        mock_nw_df = MagicMock()
        mock_nw_df.collect_schema.side_effect = Exception("Boom")
        mock_nw_df.columns = ["set_id", "node_id"]
        # Allow chaining
        mock_nw_df.select.return_value.drop_nulls.return_value.with_columns.return_value = mock_nw_df
        
        with patch("duwhal.core.ingestion.nw.from_native", return_value=mock_nw_df):
             with patch("duwhal.core.ingestion.nw.to_native", return_value=pd.DataFrame({"set_id": ["1"], "node_id": ["A"]})):
                 load_interactions(conn, pd.DataFrame({"set_id": ["1"], "node_id": ["A"]}), set_col="set_id", node_col="node_id")
                 pass

    def test_ingestion_load_matrix_exception_pass(self, conn):
        """Test load_interaction_matrix exception absorption in Pandas index logic."""
        class BadIndexDF:
            columns = ["A"] # No set_id
            def reset_index(self):
                raise ValueError("Bad index")
        
        df = BadIndexDF()
        
        with patch("duwhal.core.ingestion.nw.from_native") as mock_from_native:
            mock_nw_df = MagicMock()
            mock_nw_df.columns = ["A"] # No set_id
            mock_from_native.return_value = mock_nw_df
            
            mock_nw_df.unpivot.return_value \
                .filter.return_value \
                .select.return_value \
                .with_columns.return_value = mock_nw_df
            
            with patch("duwhal.core.ingestion.nw.to_native", return_value=pd.DataFrame({"set_id": ["1"], "node_id": ["X"]})):
                with pytest.raises(ValueError, match="Input DataFrame must have a 'set_id' column or index"):
                    load_interaction_matrix(conn, df)

    def test_ingestion_sort_callback_exception(self, conn):
        """Test load_interactions when sort_callback raises an exception."""
        df = pd.DataFrame({
            "set_id": ["O1"],
            "node_id": ["A"],
            "ts": [1]
        })
        def bad_callback(x):
            raise ValueError("Boom")
        
        # Should not raise because of try-except block in ingestion.py
        load_interactions(conn, df, sort_col="ts", sort_callback=bad_callback)
        assert conn.table_exists("interactions")

    def test_load_interaction_matrix_rename_index(self, conn):
        """Test load_interaction_matrix renaming 'index' column to 'set_id'."""
        df = pd.DataFrame({
            "item1": [1, 0],
            "item2": [0, 1]
        })
        # Resetting index on an unnamed index gives 'index' column
        load_interaction_matrix(conn, df)
        assert conn.table_exists("interactions")

    def test_describe_exception_in_append(self, conn):
        """Test DESCRIBE exception in load_interactions append mode."""
        df = pd.DataFrame({"set_id": ["S1"], "node_id": ["N1"]})
        load_interactions(conn, df)
        
        # Mocking conn.execute to fail on DESCRIBE
        original_execute = conn.execute
        def mock_execute(sql, *args, **kwargs):
            if "DESCRIBE" in sql:
                raise Exception("No describe for you")
            return original_execute(sql, *args, **kwargs)
        
        with patch.object(conn, 'execute', side_effect=mock_execute):
            load_interactions(conn, df, append=True)

    def test_check_nw_df_schema_exception(self, conn):
        """Test _check_nw_df when collect_schema raises an exception."""
        class MockNwDF:
            def collect_schema(self):
                raise Exception("Boom")
            @property
            def columns(self):
                return ["set_id", "val", "node_id"]
            def unpivot(self, *args, **kwargs):
                return self
            def filter(self, *args, **kwargs):
                return self
            def drop(self, *args, **kwargs):
                return self
            def to_native(self):
                return "dummy"
            def select(self, *args, **kwargs):
                return self
            def drop_nulls(self, *args, **kwargs):
                return self
            def with_columns(self, *args, **kwargs):
                return self

        mock_nw_df = MockNwDF()
        
        # We patch load_interactions to avoid dealing with the complex unpivot/native logic downstream
        # We only care that _resolve_set_col -> _check_nw_df -> collect_schema runs and falls back
        with patch("duwhal.core.ingestion.nw.from_native", return_value=mock_nw_df):
             with patch("duwhal.core.ingestion.load_interactions", return_value=1) as mock_load:
                 count = load_interaction_matrix(conn, pd.DataFrame({"set_id": [1]}))
                 assert count == 1
