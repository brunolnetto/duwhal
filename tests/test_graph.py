
import pytest
import pandas as pd
from duwhal.graph import InteractionGraph

class TestInteractionGraph:
    def test_context_manager(self):
        with InteractionGraph() as graph:
            assert graph.db is not None
    
    def test_load_and_rank(self, conn):
        with InteractionGraph() as graph:
            df = pd.DataFrame({
                "session": ["S1", "S1", "S2"],
                "item": ["A", "B", "A"]
            })
            graph.load_interactions(df, context_col="session", node_col="item")
            graph.build_topology(min_interactions=1)
            recs = graph.rank_nodes(seed_nodes=["A"], limit=5)
            
            # Should be a native object (Arrow Table or DataFrame)
            assert "node" in (recs.column_names if hasattr(recs, "column_names") else recs.columns)
            assert "score" in (recs.column_names if hasattr(recs, "column_names") else recs.columns)
            assert "steps" in (recs.column_names if hasattr(recs, "column_names") else recs.columns)
            assert (len(recs) if hasattr(recs, "__len__") else recs.num_rows) > 0

    def test_rank_nodes_parameters(self, conn):
        with InteractionGraph() as graph:
            df = pd.DataFrame({
                "session": ["S1", "S1", "S2"],
                "item": ["A", "B", "A"]
            })
            graph.load_interactions(df, context_col="session", node_col="item")
            graph.build_topology(min_interactions=1)
            
            # Test different scoring
            recs_freq = graph.rank_nodes(seed_nodes=["A"], scoring="frequency")
            assert len(recs_freq) > 0
            
            # Test limit
            recs_lim = graph.rank_nodes(seed_nodes=["A"], limit=1)
            assert len(recs_lim) == 1
