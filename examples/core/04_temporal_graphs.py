"""
Core example: Time-Variant Graph Algorithms.

Demonstrates all five temporal algorithms from duwhal.temporal using
a synthetic dataset with known time-shifting item clusters.
"""

from duwhal import Duwhal
from duwhal.datasets import generate_directed_sequence_data, generate_temporal_interactions
from duwhal.temporal import (
    DecayGraphRecommender,
    DirectedTemporalGraph,
    SlidingWindowGraph,
    TemporalSCCTracker,
    TemporalSnapshotDiffer,
)


def run():
    df = generate_temporal_interactions(seed=42)
    print(f"Dataset: {len(df)} rows, {df['session_id'].nunique()} sessions")
    print(f"Items: {sorted(df['item_id'].unique())}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}\n")

    with Duwhal() as dw:
        dw.load_interactions(df, set_col="session_id", node_col="item_id", sort_col="timestamp")

        # ------------------------------------------------------------------ #
        # Algorithm 1: Decay Graph Recommender                                #
        # Re-weights co-occurrences so recent events count more               #
        # ------------------------------------------------------------------ #
        print("=" * 60)
        print("1. Decay Graph Recommender (half-life = 30 days)")
        print("=" * 60)
        dgr = DecayGraphRecommender(dw.conn, half_life_days=30.0)
        recs = dgr.recommend(["A"], n=5)
        print("Recommendations for item 'A' (recency-biased):")
        print(recs.to_pandas().to_string(index=False))
        print()

        # ------------------------------------------------------------------ #
        # Algorithm 2: Sliding Window Graph                                   #
        # Snapshot of the graph as it existed in the last 28 days             #
        # ------------------------------------------------------------------ #
        print("=" * 60)
        print("2. Sliding Window Graph (window = 28 days)")
        print("=" * 60)
        swg = SlidingWindowGraph(dw, window_days=28)
        swg.fit()
        recs2 = swg.recommend(["D"], n=5)
        print("Recommendations for item 'D' (last 28 days only):")
        print(recs2.to_pandas().to_string(index=False))
        print()

        # ------------------------------------------------------------------ #
        # Algorithm 3: Temporal Snapshot Differ                               #
        # Track which edges are emerging vs fading                            #
        # ------------------------------------------------------------------ #
        print("=" * 60)
        print("3. Temporal Snapshot Differ (4 windows)")
        print("=" * 60)
        tsd = TemporalSnapshotDiffer(dw, n_windows=4)
        tsd.fit()
        print("Window labels:", tsd.window_labels)
        deltas = tsd.diff(window_a=0, window_b=1)
        print("\nTop 8 edge changes from window 1 to window 2 (|delta| sorted):")
        for d in deltas[:8]:
            print(f"  {d.source} -> {d.target}  [{d.kind}]  Δ={d.delta:+.1f}")
        tl = tsd.full_timeline()
        print(f"\nFull timeline: {len(tl)} edge-window observations")
        print()

        # ------------------------------------------------------------------ #
        # Algorithm 4: Temporal SCC Tracker                                   #
        # Watch communities form, dissolve, and evolve over time              #
        # ------------------------------------------------------------------ #
        print("=" * 60)
        print("4. Temporal SCC Tracker (4 windows)")
        print("=" * 60)
        tracker = TemporalSCCTracker(dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        for snap in tracker.timeline:
            sizes = [len(v) for v in snap.sccs.values()]
            print(f"  {snap.label}: {len(snap.sccs)} SCC(s), sizes={sizes}")
        evo = tracker.evolution_report()
        if not evo.empty:
            print("\nEvolution events:")
            print(evo[["transition", "event", "curr_members", "jaccard"]].to_string(index=False))
        print()

        # ------------------------------------------------------------------ #
        # Algorithm 5: Directed Temporal Graph                                #
        # Build A -> B edges from the order items appear in each session      #
        # ------------------------------------------------------------------ #
        print("=" * 60)
        print("5. Directed Temporal Graph (sequential patterns)")
        print("=" * 60)
        seq_df = generate_directed_sequence_data(seed=0)
        with Duwhal() as dw_seq:
            dw_seq.load_interactions(
                seq_df, set_col="session_id", node_col="item_id", sort_col="timestamp"
            )
            dtg = DirectedTemporalGraph(dw_seq.conn, min_support=2)
            dtg.build()
            edges = dtg.edge_table().to_pandas()
            print("Top directed transitions (transition probability):")
            print(edges.head(10).to_string(index=False))
            print()
            next_recs = dtg.recommend_next(["A"], n=5)
            print("What is likely to happen after 'A'?")
            print(next_recs.to_pandas().to_string(index=False))


if __name__ == "__main__":
    run()
