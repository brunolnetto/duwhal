"""Tests for DuckDB connection management."""

import pytest
from duwhal.core.connection import DuckDBConnection


class TestDuckDBConnection:

    def test_in_memory_connection(self):
        conn = DuckDBConnection()
        result = conn.execute("SELECT 42 AS answer").fetchone()
        assert result[0] == 42
        conn.close()

    def test_context_manager(self):
        with DuckDBConnection() as conn:
            result = conn.execute("SELECT 1+1").fetchone()
            assert result[0] == 2

    def test_query_returns_dataframe(self):
        with DuckDBConnection() as conn:
            arrow_table = conn.query("SELECT 1 AS a, 2 AS b")
            assert arrow_table.column_names == ["a", "b"]
            assert arrow_table.to_pylist()[0]["a"] == 1

    def test_table_exists_false(self):
        with DuckDBConnection() as conn:
            assert conn.table_exists("nonexistent_table") is False

    def test_table_exists_true(self):
        with DuckDBConnection() as conn:
            conn.execute("CREATE TABLE test_table (x INT)")
            assert conn.table_exists("test_table") is True

    def test_register_dataframe(self):
        import pandas as pd
        with DuckDBConnection() as conn:
            df = pd.DataFrame({"x": [1, 2, 3]})
            conn.register("my_df", df)
            result = conn.query("SELECT SUM(x) AS s FROM my_df")
            assert result.to_pylist()[0]["s"] == 6

    def test_memory_limit_param(self):
        # Should not raise
        conn = DuckDBConnection(memory_limit="512MB")
        conn.close()

        conn = DuckDBConnection(threads=1)
        result = conn.execute("SELECT current_setting('threads')").fetchone()
        assert str(result[0]) == "1"
        conn.close()

    def test_repr(self):
        conn = DuckDBConnection()
        assert ":memory:" in repr(conn)
        conn.close()

    def test_execute_with_params(self):
        with DuckDBConnection() as conn:
             result = conn.execute("SELECT ? AS x", [10]).fetchone()
             assert result[0] == 10

    def test_persistent_database(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        with DuckDBConnection(database=db_path) as conn:
            conn.execute("CREATE TABLE t (v INT)")
            conn.execute("INSERT INTO t VALUES (99)")

        # Re-open and verify persistence
        with DuckDBConnection(database=db_path) as conn:
            result = conn.execute("SELECT v FROM t").fetchone()
            assert result[0] == 99

from unittest.mock import MagicMock, patch
import pyarrow as pa

class TestConnectionCoverage:
    def test_connection_query_returns_table_directly(self):
        """Test query method when arrow() returns a Table directly, not a Reader."""
        with DuckDBConnection() as conn:
            # Mock the execute method to return a mock result
            mock_result = MagicMock()
            mock_table = pa.Table.from_pydict({"a": [1]})
            # arrow() returns the table directly
            mock_result.arrow.return_value = mock_table
            
            with patch.object(conn, 'execute', return_value=mock_result):
                res = conn.query("SELECT 1")
                assert isinstance(res, pa.Table)
                assert res == mock_table
