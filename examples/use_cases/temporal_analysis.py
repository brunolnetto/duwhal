"""
Use-case example: Temporal Analysis — How Graph Patterns Evolve Over Time.

Demonstrates TemporalAggregationReport and individual aggregators using the
synthetic temporal dataset. Shows how items and communities shift across
four time windows.
"""

from duwhal import Duwhal
from duwhal.aggregation import (
    CommunityAggregator,
    EdgeLifecycleAggregator,
    GraphSnapshotAggregator,
    NodeCorrelationMatrix,
    NodeTemporalAggregator,
    TemporalAggregationReport,
)
from duwhal.datasets import generate_temporal_interactions
from duwhal.temporal import TemporalSCCTracker


def analyze_temporal_graph():
    df = generate_temporal_interactions(seed=42)

    with Duwhal() as dw:
        dw.load_interactions(df, set_col="session_id", node_col="item_id", sort_col="timestamp")
        n_windows = 4

        # ------------------------------------------------------------------ #
        # Full report (runs all five aggregators in one call)                 #
        # ------------------------------------------------------------------ #
        report = TemporalAggregationReport.build(dw, n_windows=n_windows, min_cooccurrence=1)
        report.summary()

        # ------------------------------------------------------------------ #
        # 1. Node lifecycle — which items are rising vs fading?               #
        # ------------------------------------------------------------------ #
        print("\n[Detail] Node Temporal Aggregation")
        print("-" * 50)
        node_agg = NodeTemporalAggregator(dw.conn, dw.table_name, n_windows=n_windows)
        rollup = node_agg.rollup()

        rising = rollup[rollup["activity_trend"] > 0].sort_values("activity_trend", ascending=False)
        print("Rising items (positive activity trend):")
        if not rising.empty:
            print(
                rising[["node_id", "total_activity", "active_windows", "activity_trend", "status"]]
                .head(5)
                .to_string(index=False)
            )

        churned = rollup[rollup["status"] == "churned"]
        print(f"\nChurned items (stopped appearing): {churned['node_id'].tolist()}")
        emerging = rollup[rollup["status"] == "emerging"]
        print(f"Emerging items (appear mid-timeline): {emerging['node_id'].tolist()}")

        # ------------------------------------------------------------------ #
        # 2. Edge lifecycle — which connections persist vs disappear?         #
        # ------------------------------------------------------------------ #
        print("\n[Detail] Edge Lifecycle Aggregation")
        print("-" * 50)
        edge_agg = EdgeLifecycleAggregator(dw, n_windows=n_windows)
        lifecycle = edge_agg.rollup()

        persistent = lifecycle[lifecycle["edge_kind"] == "persistent"].sort_values(
            "stability", ascending=False
        )
        print(f"Persistent edges ({len(persistent)}):")
        if not persistent.empty:
            print(
                persistent[["source", "target", "stability", "weight_mean", "weight_trend"]]
                .head(5)
                .to_string(index=False)
            )

        # ------------------------------------------------------------------ #
        # 3. Graph structure per window                                       #
        # ------------------------------------------------------------------ #
        print("\n[Detail] Graph Snapshot per Window")
        print("-" * 50)
        tracker = TemporalSCCTracker(dw, n_windows=n_windows, min_cooccurrence=2)
        tracker.fit()
        graph_agg = GraphSnapshotAggregator(dw, n_windows=n_windows, scc_tracker=tracker)
        snapshots = graph_agg.compute()
        print(
            snapshots[
                ["label", "n_nodes", "n_edges", "density", "new_nodes", "churned_nodes"]
            ].to_string(index=False)
        )

        # ------------------------------------------------------------------ #
        # 4. Community cohesion per SCC per window                            #
        # ------------------------------------------------------------------ #
        print("\n[Detail] Community Cohesion")
        print("-" * 50)
        comm_agg = CommunityAggregator(dw, tracker)
        communities = comm_agg.compute()
        if not communities.empty:
            print(
                communities[
                    ["window_label", "scc_id", "size", "cohesion", "n_bridge_nodes"]
                ].to_string(index=False)
            )

        # ------------------------------------------------------------------ #
        # 5. Node synchrony — which items move together?                      #
        # ------------------------------------------------------------------ #
        print("\n[Detail] Node Activity Synchrony (top correlated pairs)")
        print("-" * 50)
        corr_agg = NodeCorrelationMatrix(dw, n_windows=n_windows, max_lag=2)
        ll = corr_agg.leading_lagging()
        if not ll.empty:
            print("Leading/lagging relationships (by |correlation|):")
            print(
                ll.head(8)[["node_a", "node_b", "lag", "correlation", "relationship"]].to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    analyze_temporal_graph()
