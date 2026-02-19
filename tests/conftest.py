
# ---------------------------------------------------------------------------
# Tests configuration
# ---------------------------------------------------------------------------

import pytest
import pandas as pd
import narwhals as nw

from duwhal.core.connection import DuckDBConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DuckDB connection for each test."""
    db = DuckDBConnection()  # :memory:
    yield db
    db.close()


@pytest.fixture
def transactions_df():
    """Standard transaction dataset matching test expectations."""
    # T1: milk, bread
    # T2: milk, bread
    # T3: milk
    # T4: milk
    # T5: bread
    # T6: milk, bread
    # T7: bread, beer, diapers
    # T8: bread, beer, diapers
    # T9: milk, bread
    # T10: beer, diapers
    data = [
        ("T1", "milk"), ("T1", "bread"),
        ("T2", "milk"), ("T2", "bread"),
        ("T3", "milk"),
        ("T4", "milk"),
        ("T5", "bread"),
        ("T6", "milk"), ("T6", "bread"),
        ("T7", "bread"), ("T7", "beer"), ("T7", "diapers"),
        ("T8", "bread"), ("T8", "beer"), ("T8", "diapers"),
        ("T9", "milk"), ("T9", "bread"),
        ("T10", "beer"), ("T10", "diapers"),
    ]
    return pd.DataFrame(data, columns=["order_id", "item_id"])


@pytest.fixture
def transactions_polars():
    """Polars version of the transaction dataset."""
    import polars as pl
    data = [
        ("T1", "milk"), ("T1", "bread"),
        ("T2", "milk"), ("T2", "bread"),
        ("T3", "milk"),
        ("T4", "milk"),
        ("T5", "bread"),
        ("T6", "milk"), ("T6", "bread"),
        ("T7", "bread"), ("T7", "beer"), ("T7", "diapers"),
        ("T8", "bread"), ("T8", "beer"), ("T8", "diapers"),
        ("T9", "milk"), ("T9", "bread"),
        ("T10", "beer"), ("T10", "diapers"),
    ]
    return pl.DataFrame(data, schema=["order_id", "item_id"], orient="row")


@pytest.fixture
def transactions_lazy(transactions_polars):
    """Lazy Polars DataFrame."""
    return transactions_polars.lazy()


@pytest.fixture
def loaded_conn(conn, transactions_df):
    """Connection with transactions data already loaded."""
    from duwhal.core.ingestion import load_interactions
    load_interactions(conn, transactions_df, set_col="order_id", node_col="item_id")
    return conn


@pytest.fixture
def duwhal_instance(transactions_df):
    """Duwhal instance with transactions loaded."""
    from duwhal.api import Duwhal
    db = Duwhal()
    db.load_interactions(transactions_df, set_col="order_id", node_col="item_id")
    yield db
    db.close()
