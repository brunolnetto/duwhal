
import pytest
import pandas as pd
from duwhal import InteractionGraph

def test_sink_scc_identification():
    # Construct a graph with two sink SCCs and some transient nodes
    # Sink 1: {A, B} - both popular
    data = [("S1", "A"), ("S1", "B")] * 10
    # Sink 2: {C, D} - both popular
    data += [("S2", "C"), ("S2", "D")] * 10
    
    # Transient E points to A but A doesn't point back (probabilistically)
    # A is in 10 sessions. E is in 1 session.
    # p(E->A) = 1.0 (high), p(A->E) = 1/10 = 0.1 (low)
    data += [("S_E", "E"), ("S_E", "A")]
    
    # Transient F points to C but C doesn't point back
    data += [("S_F", "F"), ("S_F", "C")]
    
    df = pd.DataFrame(data, columns=["session", "item"])
    
    with InteractionGraph() as graph:
        graph.load_interactions(df, context_col="session", node_col="item")
        graph.build_topology(min_interactions=1)
        
        # Use min_confidence to break symmetry
        # Total A: S1*10 + S_E = 11. 
        # Total B: S1*10 = 10.
        # cooc(A,B) = 10. p(A->B) = 10/11 = 0.9.
        # p(E->A) = 1/1 = 1.0. p(A->E) = 1/11 = 0.09.
        # Threshold min_confidence = 0.2
        sinks = graph.find_equilibrium_communities(min_cooccurrence=1, min_confidence=0.2)
        
        # Sinks should be A, B, C, D. 
        # E and F should NOT be sinks because they lead to A/C and don't have return edges.
        sink_nodes = set(sinks.column("node").to_pylist())
        assert "A" in sink_nodes
        assert "B" in sink_nodes
        assert "C" in sink_nodes
        assert "D" in sink_nodes
        assert "E" not in sink_nodes
        assert "F" not in sink_nodes
        
        # Verify SCC grouping
        scc_ids = sinks.to_pandas().groupby("members")["scc_id"].unique()
        assert len(scc_ids) == 2 # One for {A,B}, one for {C,D}

def test_sink_scc_no_edges():
    with InteractionGraph() as graph:
        df = pd.DataFrame([("S1", "A"), ("S2", "B")], columns=["s", "i"])
        graph.load_interactions(df, context_col="s", node_col="i")
        sinks = graph.find_equilibrium_communities(min_cooccurrence=1)
        assert sinks.num_rows == 0
