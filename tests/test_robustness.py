
import pytest
import pandas as pd
import polars as pl
import pyarrow as pa
from pathlib import Path
from unittest.mock import MagicMock, patch
import narwhals as nw
from duwhal import Duwhal
from duwhal.core.connection import DuckDBConnection
from duwhal.core.ingestion import load_interactions, load_interaction_matrix

def test_factory_functions(transactions_df):
    import duwhal
    db = duwhal.load(transactions_df, set_col="order_id", node_col="item_id")
    assert isinstance(db, Duwhal)
    db2 = duwhal.connect()
    assert isinstance(db2, Duwhal)

def test_ingestion_file_loading(tmp_path, conn):
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({"s": ["1"], "n": ["A"]}).to_csv(csv_file, index=False)
    load_interactions(conn, csv_file, set_col="s", node_col="n", table_name="csv_test")
    assert conn.execute("SELECT COUNT(*) FROM csv_test").fetchone()[0] == 1
    
    pq_file = tmp_path / "test.parquet"
    pd.DataFrame({"s": ["1"], "n": ["A"]}).to_parquet(pq_file)
    load_interactions(conn, pq_file, set_col="s", node_col="n", table_name="pq_test")
    assert conn.execute("SELECT COUNT(*) FROM pq_test").fetchone()[0] == 1
    
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_interactions(conn, tmp_path / "test.txt")

def test_ingestion_column_standardization(conn):
    df1 = pd.DataFrame({"s": ["1"], "n": ["A"]})
    load_interactions(conn, df1, set_col="s", node_col="n", table_name="std_test")
    
    df2 = pd.DataFrame({"s": ["2"], "n": ["B"], "time": [100]})
    load_interactions(conn, df2, set_col="s", node_col="n", sort_col="time", append=True, table_name="std_test")
    
    # Hits _check_column_exists for target too
    load_interactions(conn, df2, set_col="s", node_col="n", sort_col="time", append=True, table_name="std_test")
    
    res = conn.execute("SELECT * FROM std_test WHERE set_id='2'").fetchone()
    assert res[2] == 100 # sort_column

def test_ingestion_register_failure(conn):
    with patch("duwhal.core.connection.DuckDBConnection.register_dataframe", side_effect=RuntimeError("Serious Error")):
        with pytest.raises(RuntimeError, match="Serious Error"):
            load_interactions(conn, pd.DataFrame({"set_id":[1], "node_id":[1]}))

def test_ingestion_non_df_source(conn):
    with pytest.raises(ValueError):
        load_interaction_matrix(conn, "im definitely not a dataframe")
        
    with patch("narwhals.from_native", side_effect=Exception("Not a DF")):
        load_interactions(conn, 123)
    assert conn.table_exists("interactions")

def test_matrix_loading_index_named(conn):
    df = pd.DataFrame({"A": [1, 0], "B": [0, 1]})
    df.index.name = "set_id"
    load_interaction_matrix(conn, df, table_name="matrix_idx")
    assert conn.execute("SELECT COUNT(*) FROM matrix_idx").fetchone()[0] == 2

def test_matrix_loading_no_index_name_fallback(conn):
    df = pd.DataFrame({"A": [1, 0], "B": [0, 1]})
    load_interaction_matrix(conn, df, table_name="matrix_no_idx_name")
    assert conn.table_exists("matrix_no_idx_name")

def test_matrix_loading_reset_index_fail(conn):
    df = pd.DataFrame({"A": [1]})
    df.index.name = "session"
    with patch("pandas.DataFrame.copy", side_effect=Exception("Copy fail")):
        with pytest.raises(ValueError, match="Input DataFrame must have a 'set_id' column or index"):
            load_interaction_matrix(conn, df)

def test_matrix_unpivot_extreme_fallback(conn):
    df = pd.DataFrame({"set_id": ["S1"], "A": [1]})
    with patch("narwhals.DataFrame.unpivot", side_effect=Exception("Narwhals unpivot fail")):
        load_interaction_matrix(conn, df, table_name="unpivot_fallback")
        
        mock_df = MagicMock()
        del mock_df.melt
        mock_df.unpivot.return_value = pd.DataFrame({"set_id":["S1"], "node_id":["A"], "val":[1]})
        with patch("narwhals.DataFrame.to_native", return_value=mock_df):
             load_interaction_matrix(conn, df, table_name="unpivot_no_melt")

def test_matrix_native_filter_fallback(conn):
    df = pd.DataFrame({"set_id": ["S1"], "A": [1]})
    with patch("narwhals.DataFrame.filter", side_effect=Exception("NW filter fail")):
         load_interaction_matrix(conn, df, table_name="filter_fallback")
    
    with patch("narwhals.DataFrame.unpivot", side_effect=Exception("Fail")):
         with patch("narwhals.DataFrame.to_native", return_value=MagicMock()):
              with pytest.raises(Exception):
                   load_interaction_matrix(conn, df)

def test_recommend_auto_selection(transactions_df):
    db = Duwhal().load(transactions_df, set_col="order_id", node_col="item_id")
    res = db.recommend(strategy="auto")
    assert "recommended_item" in res.column_names
    db.fit_cf()
    res = db.recommend(strategy="auto")
    assert "recommended_item" in res.column_names
    db.association_rules()
    res = db.recommend(strategy="auto")
    assert "recommended_item" in res.column_names

def test_recommend_popular_variants(transactions_df):
    db = Duwhal().load(transactions_df, set_col="order_id", node_col="item_id")
    assert db.recommend(strategy="popular").num_rows > 0
    assert db.recommend(strategy="popular-global").num_rows > 0

def test_recommend_empty_matches(loaded_conn):
    db = Duwhal()
    db.conn = loaded_conn
    db.association_rules()
    res = db.recommend(seed_items=["IMPOSSIBLE"], strategy="rules")
    assert res.num_rows == 0

def test_recommend_invalid_strategy():
    db = Duwhal()
    with pytest.raises(ValueError, match="Unknown strategy"):
        db.recommend(strategy="magic")

def test_api_context_manager():
    with Duwhal() as db:
        assert db.conn is not None

def test_ingestion_sort_callback_error(conn):
    df = pd.DataFrame({"set_id":[1], "node_id":[1]})
    def bad_callback(d): raise RuntimeError("Bad")
    load_interactions(conn, df, sort_callback=bad_callback)

def test_metrics_edge_cases():
    from duwhal.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, average_precision
    assert precision_at_k([1], [1], 0) == 0.0
    assert ndcg_at_k([1], [1], 0) == 0.0
    assert hit_rate_at_k([1], [1], 0) == 0.0
    assert precision_at_k([], [1], 1) == 0.0
    assert recall_at_k([], [1], 1) == 0.0
    assert ndcg_at_k([1], [], 1) == 0.0
    assert average_precision([], []) == 0.0

def test_frequent_itemsets_empty(loaded_conn):
    from duwhal.mining.frequent_itemsets import FrequentItemsets
    conn = DuckDBConnection()
    conn.execute("CREATE TABLE empty (set_id VARCHAR, node_id VARCHAR)")
    fi = FrequentItemsets(conn, table_name="empty")
    res = fi.fit()
    assert res.num_rows == 0

def test_graph_recommender_unbuilt(loaded_conn):
    from duwhal.recommenders.graph import GraphRecommender
    gr = GraphRecommender(loaded_conn)
    gr.recommend(["milk"])
    assert gr._built is True
    gr2 = GraphRecommender(loaded_conn)
    gr2.get_neighbors("milk")
    assert gr2._built

def test_path_integral_features(transactions_df):
    db = Duwhal().load(transactions_df, set_col="order_id", node_col="item_id")
    res = db.recommend(["milk"], strategy="graph", scoring="path")
    assert "reason" in res.column_names
    assert "milk ->" in res.to_pylist()[0]["reason"]
    db.fit_graph(alpha=1.0)
    res_smooth = db.recommend(["milk"], strategy="graph", scoring="probability")
    assert res_smooth.num_rows > 0
    likelihood = db.score_basket(["milk", "bread"])
    assert isinstance(likelihood, float)
    assert likelihood >= 0.0

def test_association_rules_max_len_coverage(loaded_conn):
    from duwhal.mining.association_rules import AssociationRules
    df = pd.DataFrame({"set_id": ["1","1","2","2"], "node_id": ["A","B","A","B"]})
    load_interactions(loaded_conn, df, table_name="ar_test")
    ar = AssociationRules(loaded_conn, table_name="ar_test", min_support=0.1, max_antecedent_len=0)
    rules = ar.fit()
    assert rules.num_rows == 0

def test_splitting_no_set_col():
    from duwhal.evaluation.splitting import temporal_split, random_split
    df = pd.DataFrame({"A": range(10), "ts": range(10)})
    t1, t2 = temporal_split(df, "ts", test_fraction=0.2)
    assert len(t1) == 8
    assert len(t2) == 2
    r1, r2 = random_split(df, test_fraction=0.2)
    assert len(r1) == 8
    assert len(r2) == 2

def test_unpivot_fallback_error(conn):
    from duwhal.core.ingestion import _unpivot_fallback
    mock_nw = MagicMock()
    mock_native = MagicMock()
    del mock_native.melt
    del mock_native.unpivot
    mock_nw.to_native.return_value = mock_native
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        _unpivot_fallback(mock_nw, "set_id")
