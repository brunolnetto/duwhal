
import pandas as pd

from duwhal import InteractionGraph


def test_sink_scc_identification():
    # Construct a graph with two sink SCCs and some transient nodes.
    # Because load_interactions deduplicates context-item pairs, repeated rows
    # in the same context do not inflate counts.  We therefore use distinct
    # sessions to give the sink cores enough mass while keeping E/F transient.

    # Sink 1: {A, B} - 10 distinct sessions
    data = [(f"S1_{i}", "A") for i in range(10)] + [(f"S1_{i}", "B") for i in range(10)]
    # Sink 2: {C, D} - 10 distinct sessions
    data += [(f"S2_{i}", "C") for i in range(10)] + [(f"S2_{i}", "D") for i in range(10)]

    # Transient E points to A but A doesn't point back (probabilistically)
    # Total A: 10. Total E: 1. cooc(A,E) = 1. p(A->E) = 1/10 = 0.1.
    data += [("S_E", "E"), ("S_E", "A")]

    # Transient F points to C but C doesn't point back
    # Total C: 10. Total F: 1. cooc(C,F) = 1. p(C->F) = 1/10 = 0.1.
    data += [("S_F", "F"), ("S_F", "C")]

    df = pd.DataFrame(data, columns=["session", "item"])

    with InteractionGraph() as graph:
        graph.load_interactions(df, context_col="session", node_col="item")
        graph.build_topology(min_interactions=1)

        # min_confidence=0.2 drops the weak back-edges A->E and C->F, so E and
        # F end up in singleton SCCs that are not sinks (they point to A/C).
        sinks = graph.find_equilibrium_communities(min_cooccurrence=1, min_confidence=0.2)

        sink_nodes = set(sinks.column("node").to_pylist())
        assert "A" in sink_nodes
        assert "B" in sink_nodes
        assert "C" in sink_nodes
        assert "D" in sink_nodes
        assert "E" not in sink_nodes
        assert "F" not in sink_nodes

        # Verify SCC grouping
        scc_ids = sinks.to_pandas().groupby("members")["scc_id"].unique()
        assert len(scc_ids) == 2  # One for {A,B}, one for {C,D}

def test_sink_scc_no_edges():
    with InteractionGraph() as graph:
        df = pd.DataFrame([("S1", "A"), ("S2", "B")], columns=["s", "i"])
        graph.load_interactions(df, context_col="s", node_col="i")
        sinks = graph.find_equilibrium_communities(min_cooccurrence=1)
        assert sinks.num_rows == 0
