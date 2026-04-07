"""
Tests for duwhal.aggregation — five temporal aggregators + TemporalAggregationReport.
"""

from __future__ import annotations

import pandas as pd
import pytest

from duwhal.aggregation import (
    CommunityAggregator,
    EdgeLifecycleAggregator,
    GraphSnapshotAggregator,
    NodeCorrelationMatrix,
    NodeTemporalAggregator,
    TemporalAggregationReport,
)
from duwhal.api import Duwhal
from duwhal.datasets import generate_temporal_interactions
from duwhal.temporal import TemporalSCCTracker

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agg_dw():
    """Duwhal instance loaded with temporal dataset (shared across all agg tests)."""
    df = generate_temporal_interactions(seed=42, n_weeks=12)
    db = Duwhal()
    db.load_interactions(df, set_col="session_id", node_col="item_id", sort_col="timestamp")
    yield db
    db.close()


@pytest.fixture(scope="module")
def fitted_tracker(agg_dw):
    """Pre-fitted TemporalSCCTracker for tests that need SCC data."""
    tracker = TemporalSCCTracker(agg_dw, n_windows=4, min_cooccurrence=2)
    tracker.fit()
    return tracker


# ---------------------------------------------------------------------------
# 1. NodeTemporalAggregator
# ---------------------------------------------------------------------------


class TestNodeTemporalAggregator:
    def test_per_window_is_dataframe(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.per_window()
        assert isinstance(df, pd.DataFrame)

    def test_per_window_required_columns(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.per_window()
        required = {"win", "label", "node_id", "activity", "sessions", "out_degree"}
        assert required.issubset(set(df.columns))

    def test_per_window_has_n_windows_per_node(self, agg_dw):
        n = 4
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=n)
        df = agg.per_window()
        # Each win value should be between 0 and n-1
        assert df["win"].min() >= 0
        assert df["win"].max() <= n - 1

    def test_rollup_is_dataframe(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        assert isinstance(df, pd.DataFrame)

    def test_rollup_required_columns(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        required = {
            "node_id",
            "first_window",
            "last_window",
            "peak_window",
            "active_windows",
            "lifetime_windows",
            "total_activity",
            "mean_activity",
            "activity_std",
            "activity_trend",
            "total_sessions",
            "status",
        }
        assert required.issubset(set(df.columns))

    def test_rollup_status_valid_values(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        valid = {"active", "churned", "emerging", "absent"}
        assert set(df["status"].unique()).issubset(valid)

    def test_rollup_activity_nonnegative(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        assert (df["total_activity"] >= 0).all()
        assert (df["active_windows"] >= 0).all()

    def test_rollup_sorted_by_total_activity(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        acts = df["total_activity"].tolist()
        assert acts == sorted(acts, reverse=True)

    def test_known_items_in_rollup(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        assert "X" in df["node_id"].values  # bridge node always present

    def test_emerging_nodes_detected(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        emerging = df[df["status"] == "emerging"]["node_id"].tolist()
        # Late-cluster items G, H, I are expected to emerge
        assert any(item in emerging for item in ["G", "H", "I"])

    def test_churned_nodes_detected(self, agg_dw):
        agg = NodeTemporalAggregator(agg_dw.conn, agg_dw.table_name, n_windows=4)
        df = agg.rollup()
        churned = df[df["status"] == "churned"]["node_id"].tolist()
        # Early-cluster items A, B, C are expected to churn
        assert any(item in churned for item in ["A", "B", "C"])


# ---------------------------------------------------------------------------
# 2. EdgeLifecycleAggregator
# ---------------------------------------------------------------------------


class TestEdgeLifecycleAggregator:
    def test_per_window_delegates(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.per_window()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"window", "label", "source", "target", "weight"}

    def test_rollup_is_dataframe(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.rollup()
        assert isinstance(df, pd.DataFrame)

    def test_rollup_required_columns(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.rollup()
        required = {
            "source",
            "target",
            "active_windows",
            "stability",
            "weight_mean",
            "weight_std",
            "weight_max",
            "weight_trend",
            "peak_window",
            "first_window",
            "last_window",
            "edge_kind",
        }
        assert required.issubset(set(df.columns))

    def test_rollup_edge_kind_valid(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.rollup()
        valid = {"persistent", "transient", "sporadic"}
        assert set(df["edge_kind"].unique()).issubset(valid)

    def test_stability_range(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.rollup()
        assert (df["stability"] >= 0.0).all()
        assert (df["stability"] <= 1.0 + 1e-9).all()

    def test_persistent_edges_appear_in_most_windows(self, agg_dw):
        n = 4
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=n)
        df = agg.rollup()
        persistent = df[df["edge_kind"] == "persistent"]
        if not persistent.empty:
            assert (persistent["active_windows"] > n * 0.5).all()

    def test_transient_edges_appear_once(self, agg_dw):
        agg = EdgeLifecycleAggregator(agg_dw, n_windows=4)
        df = agg.rollup()
        transient = df[df["edge_kind"] == "transient"]
        if not transient.empty:
            assert (transient["active_windows"] == 1).all()


# ---------------------------------------------------------------------------
# 3. GraphSnapshotAggregator
# ---------------------------------------------------------------------------


class TestGraphSnapshotAggregator:
    def test_compute_is_dataframe(self, agg_dw):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4)
        df = agg.compute()
        assert isinstance(df, pd.DataFrame)

    def test_compute_has_n_windows_rows(self, agg_dw):
        n = 4
        agg = GraphSnapshotAggregator(agg_dw, n_windows=n)
        df = agg.compute()
        assert len(df) == n

    def test_compute_required_columns(self, agg_dw):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4)
        df = agg.compute()
        required = {"win", "label", "n_nodes", "n_edges", "density", "n_sessions"}
        assert required.issubset(set(df.columns))

    def test_density_range(self, agg_dw):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4)
        df = agg.compute()
        valid = df["density"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_n_nodes_positive(self, agg_dw):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4)
        df = agg.compute()
        assert (df["n_nodes"] > 0).all()

    def test_with_scc_tracker(self, agg_dw, fitted_tracker):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4, scc_tracker=fitted_tracker)
        df = agg.compute()
        assert "n_sccs" in df.columns
        assert "scc_coverage" in df.columns

    def test_scc_coverage_range(self, agg_dw, fitted_tracker):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4, scc_tracker=fitted_tracker)
        df = agg.compute()
        cov = df["scc_coverage"].dropna()
        assert (cov >= 0.0).all()
        assert (cov <= 1.0 + 1e-9).all()

    def test_without_scc_tracker(self, agg_dw):
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4, scc_tracker=None)
        df = agg.compute()
        assert df["n_sccs"].isna().all()

    def test_node_conservation(self, agg_dw):
        """n_nodes[i] == n_nodes[i-1] + new_nodes[i] - churned_nodes[i]."""
        agg = GraphSnapshotAggregator(agg_dw, n_windows=4)
        df = agg.compute().reset_index(drop=True)
        for i in range(1, len(df)):
            expected = (
                df.loc[i - 1, "n_nodes"] + df.loc[i, "new_nodes"] - df.loc[i, "churned_nodes"]
            )
            assert df.loc[i, "n_nodes"] == expected, (
                f"Window {i}: expected n_nodes={expected}, got {df.loc[i, 'n_nodes']}"
            )


# ---------------------------------------------------------------------------
# 4. CommunityAggregator
# ---------------------------------------------------------------------------


class TestCommunityAggregator:
    def test_compute_is_dataframe(self, agg_dw, fitted_tracker):
        agg = CommunityAggregator(agg_dw, fitted_tracker)
        df = agg.compute()
        assert isinstance(df, pd.DataFrame)

    def test_compute_required_columns(self, agg_dw, fitted_tracker):
        agg = CommunityAggregator(agg_dw, fitted_tracker)
        df = agg.compute()
        required = {
            "window",
            "window_label",
            "scc_id",
            "size",
            "members",
            "cohesion",
            "bridge_nodes",
            "n_bridge_nodes",
        }
        assert required.issubset(set(df.columns))

    def test_cohesion_range(self, agg_dw, fitted_tracker):
        agg = CommunityAggregator(agg_dw, fitted_tracker)
        df = agg.compute()
        assert (df["cohesion"] >= 0.0).all()

    def test_size_positive(self, agg_dw, fitted_tracker):
        agg = CommunityAggregator(agg_dw, fitted_tracker)
        df = agg.compute()
        if not df.empty:
            assert (df["size"] > 0).all()

    def test_n_bridge_nodes_nonnegative(self, agg_dw, fitted_tracker):
        agg = CommunityAggregator(agg_dw, fitted_tracker)
        df = agg.compute()
        assert (df["n_bridge_nodes"] >= 0).all()


# ---------------------------------------------------------------------------
# 5. NodeCorrelationMatrix
# ---------------------------------------------------------------------------


class TestNodeCorrelationMatrix:
    def test_synchrony_is_dataframe(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4)
        df = corr.synchrony()
        assert isinstance(df, pd.DataFrame)

    def test_synchrony_is_square(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4)
        df = corr.synchrony()
        assert df.shape[0] == df.shape[1]

    def test_synchrony_diagonal_is_one(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4)
        df = corr.synchrony()
        import numpy as np

        diag = df.values.diagonal()
        assert all(abs(v - 1.0) < 1e-6 for v in diag if not np.isnan(v))

    def test_synchrony_range(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4)
        df = corr.synchrony()
        import numpy as np

        vals = df.values[~np.isnan(df.values)]
        assert (vals >= -1.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_leading_lagging_is_dataframe(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4, max_lag=2)
        df = corr.leading_lagging()
        assert isinstance(df, pd.DataFrame)

    def test_leading_lagging_columns(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4, max_lag=2)
        df = corr.leading_lagging()
        if not df.empty:
            required = {"node_a", "node_b", "lag", "correlation", "relationship"}
            assert required.issubset(set(df.columns))

    def test_leading_lagging_correlation_range(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4, max_lag=2)
        df = corr.leading_lagging()
        if not df.empty:
            assert (df["correlation"] >= -1.0 - 1e-9).all()
            assert (df["correlation"] <= 1.0 + 1e-9).all()

    def test_leading_lagging_lag_positive(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4, max_lag=2)
        df = corr.leading_lagging()
        if not df.empty:
            assert (df["lag"] >= 1).all()

    def test_leading_lagging_sorted_by_abs_correlation(self, agg_dw):
        corr = NodeCorrelationMatrix(agg_dw, n_windows=4, max_lag=2)
        df = corr.leading_lagging()
        if len(df) > 1:
            abs_corrs = df["correlation"].abs().tolist()
            assert abs_corrs == sorted(abs_corrs, reverse=True)


# ---------------------------------------------------------------------------
# Composite – TemporalAggregationReport
# ---------------------------------------------------------------------------


class TestTemporalAggregationReport:
    @pytest.fixture(scope="class")
    def report(self, agg_dw):
        return TemporalAggregationReport.build(agg_dw, n_windows=4, min_cooccurrence=1, max_lag=2)

    def test_report_is_dataclass(self, report):
        assert isinstance(report, TemporalAggregationReport)

    def test_all_fields_are_dataframes(self, report):
        df_fields = [
            "node_per_window",
            "node_rollup",
            "edge_per_window",
            "edge_lifecycle",
            "graph_snapshots",
            "community_metrics",
            "synchrony_matrix",
            "leading_lagging",
        ]
        for field in df_fields:
            val = getattr(report, field)
            assert isinstance(val, pd.DataFrame), f"{field} should be a DataFrame"
            assert not val.empty, f"{field} should not be empty"

    def test_window_labels_list(self, report):
        assert isinstance(report.window_labels, list)
        assert len(report.window_labels) == 4

    def test_graph_snapshots_has_4_rows(self, report):
        assert len(report.graph_snapshots) == 4

    def test_node_rollup_has_status_column(self, report):
        assert "status" in report.node_rollup.columns

    def test_edge_lifecycle_has_edge_kind(self, report):
        assert "edge_kind" in report.edge_lifecycle.columns

    def test_summary_runs_without_error(self, report, capsys):
        report.summary()
        captured = capsys.readouterr()
        assert "TEMPORAL AGGREGATION REPORT" in captured.out

    def test_summary_contains_sections(self, report, capsys):
        report.summary()
        out = capsys.readouterr().out
        assert "Graph Snapshot Evolution" in out
        assert "Node Lifecycle Highlights" in out
        assert "Edge Lifecycle" in out
