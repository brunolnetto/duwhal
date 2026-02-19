"""Tests for sequential pattern mining."""

import pytest
import pandas as pd
import pyarrow as pa
from datetime import datetime, timedelta

from duwhal.core.connection import DuckDBConnection
from duwhal.mining.sequences import SequentialPatterns


@pytest.fixture
def sequential_conn():
    """Transactions with a timestamp column for sequential mining."""
    conn = DuckDBConnection()
    base = datetime(2024, 1, 1, 10, 0, 0)
    rows = [
        # T1: A → B → C
        ("T1", "A", base),
        ("T1", "B", base + timedelta(minutes=5)),
        ("T1", "C", base + timedelta(minutes=10)),
        # T2: A → B
        ("T2", "A", base),
        ("T2", "B", base + timedelta(minutes=3)),
        # T3: A → C
        ("T3", "A", base),
        ("T3", "C", base + timedelta(minutes=7)),
        # T4: B → C
        ("T4", "B", base),
        ("T4", "C", base + timedelta(minutes=4)),
        # T5: A → B → C
        ("T5", "A", base),
        ("T5", "B", base + timedelta(minutes=2)),
        ("T5", "C", base + timedelta(minutes=8)),
    ]
    df = pd.DataFrame(rows, columns=["order_id", "item_id", "event_time"])
    conn.register("_tmp", df)
    conn.execute("""
        CREATE TABLE interactions AS
        SELECT order_id AS set_id, item_id AS node_id, event_time::TIMESTAMP AS event_time FROM _tmp
    """)
    yield conn
    conn.close()


class TestSequentialPatterns:

    def test_returns_dataframe(self, sequential_conn):
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.1)
        result = sp.fit()
        assert isinstance(result, pa.Table)

    def test_required_columns(self, sequential_conn):
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.1)
        result = sp.fit()
        assert set(result.column_names) >= {"prefix", "suffix", "support", "count"}

    def test_a_then_b_frequent(self, sequential_conn):
        # A→B appears in T1, T2, T5 → support = 3/5 = 0.6
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.1)
        result = sp.fit()
        rows = result.to_pylist()
        row = next((r for r in rows if r["prefix"] == "A" and r["suffix"] == "B"), None)
        assert row is not None
        assert abs(row["support"] - 0.6) < 1e-9

    def test_min_support_filters(self, sequential_conn):
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.5)
        result = sp.fit()
        if result.num_rows > 0:
            for val in result.column("support"):
                assert val.as_py() >= 0.5

    def test_missing_timestamp_raises(self, loaded_conn):
        sp = SequentialPatterns(loaded_conn, timestamp_col="nonexistent_col", min_support=0.1)
        with pytest.raises(ValueError, match="not found"):
            sp.fit()

    def test_support_between_0_and_1(self, sequential_conn):
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.05)
        result = sp.fit()
        if result.num_rows > 0:
            for val in result.column("support"):
                assert 0 < val.as_py() <= 1

    def test_b_not_before_a(self, sequential_conn):
        # B→A should not appear because A always comes first within sessions
        sp = SequentialPatterns(sequential_conn, timestamp_col="event_time", min_support=0.1)
        result = sp.fit()
        rows = result.to_pylist()
        ba = [r for r in rows if r["prefix"] == "B" and r["suffix"] == "A"]
        assert len(ba) == 0

    def test_sequential_patterns_sort_fallback(self, conn):
        """Test fallback to sort_column if event_time is missing."""
        from duwhal.core.ingestion import load_interactions
        df = pd.DataFrame({
            "set_id": ["S1", "S1"],
            "node_id": ["A", "B"],
            "sort_column": [1, 2]
        })
        load_interactions(conn, df, sort_col="sort_column")
        # Should fallback to sort_column because event_time is missing
        sp = SequentialPatterns(conn, min_support=0.01)
        patterns = sp.fit()
        assert len(patterns) > 0
