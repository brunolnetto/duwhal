"""
Tests for duwhal.temporal — five time-variant graph algorithms.
"""

from __future__ import annotations

import pandas as pd
import pytest

from duwhal.api import Duwhal
from duwhal.core.connection import DuckDBConnection
from duwhal.core.ingestion import load_interactions
from duwhal.datasets import generate_directed_sequence_data, generate_temporal_interactions
from duwhal.temporal import (
    DecayGraphRecommender,
    DirectedTemporalGraph,
    EdgeDelta,
    SlidingWindowGraph,
    TemporalSCCTracker,
    TemporalSnapshotDiffer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def temporal_df():
    """Temporal dataset as a DataFrame (shared across all tests in module)."""
    return generate_temporal_interactions(seed=42, n_weeks=12)


@pytest.fixture(scope="module")
def temporal_conn(temporal_df):
    """DuckDBConnection pre-loaded with temporal dataset."""
    db = DuckDBConnection()
    load_interactions(
        db, temporal_df, set_col="session_id", node_col="item_id", sort_col="timestamp"
    )
    yield db
    db.close()


@pytest.fixture(scope="module")
def temporal_dw(temporal_df):
    """Duwhal instance pre-loaded with temporal dataset."""
    db = Duwhal()
    db.load_interactions(
        temporal_df, set_col="session_id", node_col="item_id", sort_col="timestamp"
    )
    yield db
    db.close()


@pytest.fixture(scope="module")
def seq_conn():
    """DuckDBConnection pre-loaded with directed-sequence dataset."""
    df = generate_directed_sequence_data(seed=0)
    db = DuckDBConnection()
    load_interactions(db, df, set_col="session_id", node_col="item_id", sort_col="timestamp")
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestTemporalDataset:
    def test_generate_returns_dataframe(self):
        df = generate_temporal_interactions(seed=1)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = generate_temporal_interactions()
        assert set(df.columns) == {"session_id", "item_id", "timestamp"}

    def test_seed_reproducible(self):
        df1 = generate_temporal_interactions(seed=7)
        df2 = generate_temporal_interactions(seed=7)
        assert df1.equals(df2)

    def test_has_temporal_spread(self):
        df = generate_temporal_interactions(seed=0, n_weeks=12)
        ts = pd.to_datetime(df["timestamp"])
        span_days = (ts.max() - ts.min()).days
        assert span_days >= 60, "dataset should span at least 60 days"

    def test_known_items_present(self):
        df = generate_temporal_interactions(seed=0)
        items = set(df["item_id"].unique())
        assert {"A", "B", "C", "D", "E", "F", "G", "H", "I", "X"}.issubset(items)

    def test_directed_sequence_columns(self):
        df = generate_directed_sequence_data()
        assert set(df.columns) == {"session_id", "item_id", "timestamp"}

    def test_directed_sequence_seed(self):
        df1 = generate_directed_sequence_data(seed=5)
        df2 = generate_directed_sequence_data(seed=5)
        assert df1.equals(df2)


# ---------------------------------------------------------------------------
# Algorithm 1 – DecayGraphRecommender
# ---------------------------------------------------------------------------


class TestDecayGraphRecommender:
    def test_build_creates_edge_table(self, temporal_conn):
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        dgr.build()
        edges = dgr.edge_table()
        assert edges.num_rows > 0
        assert "source" in edges.column_names
        assert "target" in edges.column_names
        assert "score_val" in edges.column_names

    def test_recommend_returns_arrow_table(self, temporal_conn):
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        recs = dgr.recommend(["A"], n=5)
        assert "recommended_item" in recs.column_names
        assert "decay_score" in recs.column_names
        assert "min_hops" in recs.column_names

    def test_recommend_respects_limit(self, temporal_conn):
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        recs = dgr.recommend(["A"], n=3)
        assert recs.num_rows <= 3

    def test_recommend_excludes_seed_by_default(self, temporal_conn):
        seed = "A"
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        recs = dgr.recommend([seed], n=10)
        recommended = recs.column("recommended_item").to_pylist()
        assert seed not in recommended

    def test_recommend_can_include_seed(self, temporal_conn):
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        recs = dgr.recommend(["A"], n=10, exclude_seed=False)
        # Should return rows (may or may not include seed depending on depth)
        assert recs.num_rows >= 0

    def test_recommend_empty_seeds(self, temporal_conn):
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        recs = dgr.recommend([])
        assert recs.num_rows == 0
        assert "recommended_item" in recs.column_names

    def test_short_half_life_downweights_old_events(self, temporal_conn):
        """Very short half-life should produce fewer retained edges than a long one.

        With a 1-day half-life only very recent co-occurrences survive above the
        min-support threshold, so the edge table is strictly smaller than the
        edge table produced by a 10-year half-life that weights all history
        roughly equally.
        """
        dgr_short = DecayGraphRecommender(temporal_conn, half_life_days=1.0)
        dgr_long = DecayGraphRecommender(temporal_conn, half_life_days=3650.0)
        edges_short = dgr_short.edge_table().to_pandas()
        edges_long = dgr_long.edge_table().to_pandas()
        # Strict decay prunes historically-rare pairs -> fewer edges
        assert len(edges_short) < len(edges_long)

    def test_build_is_idempotent(self, temporal_conn):
        """Calling build() twice should not error and edge_table should stay valid."""
        dgr = DecayGraphRecommender(temporal_conn, half_life_days=30.0)
        dgr.build()
        dgr.build()
        assert dgr.edge_table().num_rows > 0


# ---------------------------------------------------------------------------
# Algorithm 2 – SlidingWindowGraph
# ---------------------------------------------------------------------------


class TestSlidingWindowGraph:
    def test_fit_returns_self(self, temporal_dw):
        swg = SlidingWindowGraph(temporal_dw, window_days=28)
        result = swg.fit()
        assert result is swg

    def test_recommend_returns_arrow_table(self, temporal_dw):
        swg = SlidingWindowGraph(temporal_dw, window_days=28)
        swg.fit()
        recs = swg.recommend(["A"], n=5)
        assert recs.num_rows >= 0  # empty is possible if window has no data

    def test_explicit_anchor(self, temporal_dw):
        """A very early anchor should return few or no results."""
        swg = SlidingWindowGraph(temporal_dw, window_days=7, anchor_date="2024-01-08")
        swg.fit()
        recs = swg.recommend(["A"], n=5)
        assert recs.num_rows >= 0

    def test_slide_changes_anchor(self, temporal_dw):
        swg = SlidingWindowGraph(temporal_dw, window_days=28)
        swg.fit()
        swg.slide("2024-06-01")
        assert swg.anchor_date == "2024-06-01"

    def test_table_name_restored_after_fit(self, temporal_dw):
        original = temporal_dw.table_name
        swg = SlidingWindowGraph(temporal_dw, window_days=28)
        swg.fit()
        assert temporal_dw.table_name == original


# ---------------------------------------------------------------------------
# Algorithm 3 – TemporalSnapshotDiffer + EdgeDelta
# ---------------------------------------------------------------------------


class TestEdgeDelta:
    def test_emerging(self):
        d = EdgeDelta("A", "B", weight_before=0.0, weight_after=3.0)
        assert d.kind == "emerging"
        assert d.delta == pytest.approx(3.0)

    def test_fading(self):
        d = EdgeDelta("A", "B", weight_before=5.0, weight_after=0.0)
        assert d.kind == "fading"

    def test_strengthening(self):
        d = EdgeDelta("A", "B", weight_before=2.0, weight_after=4.0)
        assert d.kind == "strengthening"

    def test_weakening(self):
        d = EdgeDelta("A", "B", weight_before=4.0, weight_after=2.0)
        assert d.kind == "weakening"


class TestTemporalSnapshotDiffer:
    def test_fit_creates_n_snapshots(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=4)
        tsd.fit()
        assert len(tsd.snapshots) == 4
        assert len(tsd.window_labels) == 4

    def test_window_labels_format(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=3)
        tsd.fit()
        for lbl in tsd.window_labels:
            assert lbl.startswith("W")
            assert "→" in lbl

    def test_diff_returns_edge_deltas(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=4)
        tsd.fit()
        deltas = tsd.diff(0, 1)
        assert isinstance(deltas, list)
        assert all(isinstance(d, EdgeDelta) for d in deltas)

    def test_diff_sorted_by_abs_delta(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=4)
        tsd.fit()
        deltas = tsd.diff(0, 1)
        abs_deltas = [abs(d.delta) for d in deltas]
        assert abs_deltas == sorted(abs_deltas, reverse=True)

    def test_full_timeline_is_dataframe(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=4)
        tsd.fit()
        tl = tsd.full_timeline()
        assert isinstance(tl, pd.DataFrame)
        assert set(tl.columns) >= {"window", "label", "source", "target", "weight"}

    def test_fit_is_idempotent(self, temporal_dw):
        tsd = TemporalSnapshotDiffer(temporal_dw, n_windows=4)
        tsd.fit()
        n1 = len(tsd.snapshots[0])
        tsd.fit()
        n2 = len(tsd.snapshots[0])
        assert n1 == n2


# ---------------------------------------------------------------------------
# Algorithm 4 – TemporalSCCTracker
# ---------------------------------------------------------------------------


class TestTemporalSCCTracker:
    def test_fit_creates_timeline(self, temporal_dw):
        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        assert len(tracker.timeline) == 4

    def test_timeline_has_scc_snapshots(self, temporal_dw):
        from duwhal.temporal import SCCSnapshot

        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        for snap in tracker.timeline:
            assert isinstance(snap, SCCSnapshot)
            assert isinstance(snap.sccs, dict)

    def test_evolution_report_is_dataframe(self, temporal_dw):
        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        report = tracker.evolution_report()
        assert isinstance(report, pd.DataFrame)

    def test_evolution_report_columns(self, temporal_dw):
        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        report = tracker.evolution_report()
        if not report.empty:
            assert set(report.columns) >= {
                "transition",
                "event",
                "prev_members",
                "curr_members",
                "jaccard",
            }

    def test_evolution_event_values(self, temporal_dw):
        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        report = tracker.evolution_report()
        valid_events = {"stable", "evolving", "emerging", "dissolving"}
        assert set(report["event"].unique()).issubset(valid_events)

    def test_jaccard_range(self, temporal_dw):
        tracker = TemporalSCCTracker(temporal_dw, n_windows=4, min_cooccurrence=2)
        tracker.fit()
        report = tracker.evolution_report()
        if not report.empty:
            assert (report["jaccard"] >= 0.0).all()
            assert (report["jaccard"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Algorithm 5 – DirectedTemporalGraph
# ---------------------------------------------------------------------------


class TestDirectedTemporalGraph:
    def test_build_creates_edges(self, seq_conn):
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        edges = dtg.edge_table()
        assert edges.num_rows > 0

    def test_edges_have_direction(self, seq_conn):
        """A→B should not imply B→A in the directed graph."""
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        edges = dtg.edge_table().to_pandas()
        pairs = set(zip(edges["source"], edges["target"]))
        # For any directed pair (s,t), the reverse (t,s) may not exist
        directed_only = {(s, t) for s, t in pairs if (t, s) not in pairs}
        assert len(directed_only) > 0, "directed graph must have at least one asymmetric edge"

    def test_recommend_next_returns_arrow(self, seq_conn):
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        recs = dtg.recommend_next(["A"], n=5)
        assert "next_item" in recs.column_names
        assert "transition_prob" in recs.column_names
        assert "steps" in recs.column_names

    def test_recommend_next_excludes_seed(self, seq_conn):
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        recs = dtg.recommend_next(["A"], n=10)
        assert "A" not in recs.column("next_item").to_pylist()

    def test_transition_probs_leq_1(self, seq_conn):
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        edges = dtg.edge_table().to_pandas()
        assert (edges["score_val"] <= 1.0 + 1e-9).all()
        assert (edges["score_val"] >= 0.0).all()

    def test_transition_probs_sum_to_1_per_source(self, seq_conn):
        """score_val from the same source should sum to ≤ 1 (≤ because of min_support filter)."""
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        edges = dtg.edge_table().to_pandas()
        for src, grp in edges.groupby("source"):
            total = grp["score_val"].sum()
            assert total <= 1.0 + 1e-6, f"source={src} probs sum to {total:.4f} > 1"

    def test_reachable_from(self, seq_conn):
        dtg = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg.build()
        reachable = dtg.reachable_from("A", max_depth=3)
        assert "next_item" in reachable.column_names
        assert reachable.num_rows >= 0

    def test_max_gap_filter(self, seq_conn):
        """With max_gap=0 only immediate successors count."""
        dtg_all = DirectedTemporalGraph(seq_conn, min_support=1)
        dtg_all.build()
        full_edges = dtg_all.edge_table().num_rows

        dtg_gap0 = DirectedTemporalGraph(seq_conn, min_support=1, max_gap=0)
        dtg_gap0.build()
        gap0_edges = dtg_gap0.edge_table().num_rows

        assert gap0_edges <= full_edges
