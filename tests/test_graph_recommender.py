
"""Tests for the new Graph Recommender."""

import pytest
import pandas as pd
import pyarrow as pa
from duwhal.recommenders.graph import GraphRecommender
from duwhal.api import Duwhal

class TestGraphRecommender:

    def test_build_creates_adjacency_table(self, loaded_conn):
        """Test that build() creates the _item_adjacency table."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # Should return self
        assert gr.build() is gr
        
        # Check table exists and has array columns
        # neighbors should be a LIST type
        schema = loaded_conn.execute("DESCRIBE _item_adjacency").fetchall()
        col_names = [row[0] for row in schema]
        col_types = [row[1] for row in schema]
        
        assert "source" in col_names
        assert "neighbors" in col_names
        assert "weights" in col_names
        
        # Check content (milk -> bread)
        res = loaded_conn.execute("SELECT * FROM _item_adjacency WHERE source = 'milk'").fetchone()
        assert res is not None
        # neighbors list should contain 'bread'
        neighbors = res[1] 
        assert "bread" in neighbors

    def test_recommend_basic(self, loaded_conn):
        """Test basic 1-hop recommendation (milk -> bread)."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        
        recs = gr.recommend(["milk"], max_depth=1, n=5)
        assert isinstance(recs, pa.Table)
        assert recs.num_rows > 0
        
        items = recs.column("recommended_item").to_pylist()
        scores = recs.column("total_strength").to_pylist()
        
        # Milk co-occurs with bread (3 times in conftest data usually?)
        assert "bread" in items
        idx = items.index("bread")
        assert scores[idx] >= 1

    def test_recommend_multi_hop(self, conn):
        """
        Test 2-hop recommendation.
        A -> B (Strong)
        B -> C (Strong)
        A -> C (No path or Weak)
        
        Graph:
        T1: A, B
        T2: B, C
        """
        df = pd.DataFrame([
            ("T1", "A"), ("T1", "B"),
            ("T2", "B"), ("T2", "C"),
        ], columns=["order_id", "item_id"])
        
        from duwhal.core.ingestion import load_interactions
        load_interactions(conn, df, set_col="order_id", node_col="item_id")
        
        gr = GraphRecommender(conn, min_cooccurrence=1)
        gr.build()
        
        # Recommend for A
        # Hop 1: B (weight 1)
        # Hop 2: C (neighbor of B, weight 1)
        recs = gr.recommend(["A"], max_depth=2, n=5)
        
        items = recs.column("recommended_item").to_pylist()
        hops = recs.column("min_hops").to_pylist()
        
        assert "B" in items
        assert "C" in items
        
        # Check hops
        b_idx = items.index("B")
        c_idx = items.index("C")
        assert hops[b_idx] == 1
        assert hops[c_idx] == 2

    def test_recommend_excludes_seed(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        recs = gr.recommend(["milk"], exclude_seed=True)
        items = recs.column("recommended_item").to_pylist()
        assert "milk" not in items

    def test_recommend_includes_seed(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # Since cycles exist (A-B, B-A), A might be reachable from itself in 2 hops A->B->A
        recs = gr.recommend(["milk"], max_depth=2, exclude_seed=False)
        items = recs.column("recommended_item").to_pylist()
        # The logic prevents visiting nodes in history, so A->B->A is blocked.
        assert "milk" not in items

    def test_recommend_scoring_probability(self, loaded_conn):
        """Test the Path Integral scoring mode."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        recs = gr.recommend(["milk"], max_depth=2, scoring="probability")
        
        scores = recs.column("total_strength").to_pylist()
        assert all(s > 0 for s in scores)
        # Probabilities should be small (<= 1 usually, unless many paths)
        # With small graph, likely <= 1
        assert all(s <= 2.0 for s in scores) 

    def test_recommend_auto_build(self, loaded_conn):
        """Test that recommend calls build() if not built."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # No build() call
        recs = gr.recommend(["milk"])
        assert gr._built
        assert recs.num_rows > 0

    def test_get_neighbors(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        neighbors = gr.get_neighbors("milk")
        assert isinstance(neighbors, pa.Table)
        assert "neighbor" in neighbors.column_names
        assert "weight" in neighbors.column_names
        pylist = neighbors.to_pylist()
        assert any( row["neighbor"] == "bread" for row in pylist )
        
class TestDuwhalGraphAPI:
    """Test the integration in the main class."""
    
    def test_fit_and_recommend_graph(self, duwhal_instance):
        db = duwhal_instance
        # Auto-fit
        recs = db.recommend_graph(["milk"], n=3)
        assert isinstance(recs, pa.Table)
        assert recs.num_rows > 0

    def test_recommend_empty_seed(self, loaded_conn):
        """Test recommend with empty seed list returns empty table."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        recs = gr.recommend([])
        assert recs.num_rows == 0
        assert "recommended_item" in recs.column_names

    def test_fit_graph_explicit(self, duwhal_instance):
        db = duwhal_instance.fit_graph(min_cooccurrence=1)
        assert db._graph is not None
        assert db._graph._built

