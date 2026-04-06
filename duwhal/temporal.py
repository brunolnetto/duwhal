"""
duwhal.temporal
===============
Time-variant graph algorithms built on top of duwhal's graph infrastructure.

Five algorithms exploit duwhal's graph bias to handle temporally evolving data:

1. DecayGraphRecommender   — Edge weights decay exponentially with age
2. SlidingWindowGraph      — Rolling window re-builds graph on recency
3. TemporalSnapshotDiffer  — Detects emerging / fading edges across windows
4. TemporalSCCTracker      — Tracks SCC formation, persistence, dissolution
5. DirectedTemporalGraph   — Sequential-pattern graph: directed, time-ordered edges
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.mining.sink_sccs import SinkSCCFinder

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Algorithm 1 – Decay Graph Recommender
# ---------------------------------------------------------------------------


class DecayGraphRecommender:
    """
    Replaces duwhal's raw co-occurrence count with an exponentially
    time-decayed weight::

        w(u, v, t) = Σ  exp(-λ · Δdays)   for each co-occurrence at time t

    This makes the graph "remember" the recent past more strongly.
    The decay half-life λ is expressed in days.

    Parameters
    ----------
    conn : DuckDBConnection
        Live duwhal connection.
    table_name : str
        Name of the interactions table (must have sort_column).
    half_life_days : float
        Edge weight halves every ``half_life_days`` days.
    reference_date : str, optional
        ISO date string used as "now". Defaults to MAX(sort_column).
    min_weight : float
        Minimum decayed weight for an edge to be kept.
    """

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        half_life_days: float = 30.0,
        reference_date: Optional[str] = None,
        min_weight: float = 0.01,
    ):
        self.conn = conn
        self.table_name = table_name
        self.lam = math.log(2) / half_life_days  # λ = ln2 / half-life
        self.reference_date = reference_date
        self.min_weight = min_weight
        self._built = False

    def build(self) -> DecayGraphRecommender:
        ref = (
            f"TIMESTAMP '{self.reference_date}'"
            if self.reference_date
            else f"(SELECT MAX(sort_column::TIMESTAMP) FROM {self.table_name})"
        )

        # Decayed total interactions per node (denominator for probability)
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _decay_node_totals AS
            SELECT
                node_id,
                SUM(EXP({-self.lam} * DATE_DIFF('day', sort_column::TIMESTAMP, {ref}))) AS decay_total
            FROM {self.table_name}
            GROUP BY node_id
        """)

        # Decayed co-occurrence weight for each directed edge (a → b)
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _decay_edges AS
            SELECT
                a.node_id AS source,
                b.node_id AS target,
                SUM(EXP({-self.lam} * DATE_DIFF('day', a.sort_column::TIMESTAMP, {ref}))) AS decay_weight
            FROM {self.table_name} a
            JOIN {self.table_name} b
                ON  a.set_id  = b.set_id
                AND a.node_id != b.node_id
            GROUP BY source, target
            HAVING decay_weight >= {self.min_weight}
        """)

        # Scored edge (transition probability with decayed weights)
        self.conn.execute("""
            CREATE OR REPLACE TEMP TABLE _decay_edges_scored AS
            SELECT
                e.source,
                e.target,
                e.decay_weight,
                e.decay_weight / t.decay_total AS score_val
            FROM _decay_edges e
            JOIN _decay_node_totals t ON e.source = t.node_id
        """)

        self._built = True
        return self

    def recommend(
        self,
        seed_items: list[str],
        max_depth: int = 2,
        n: int = 10,
        exclude_seed: bool = True,
    ) -> pa.Table:
        if not self._built:
            self.build()
        if not seed_items:
            return pa.table(
                {"recommended_item": [], "decay_score": [], "min_hops": [], "reason": []}
            )

        seeds_sql = ", ".join(f"'{s}'" for s in seed_items)
        exc = f"AND item NOT IN ({seeds_sql})" if exclude_seed else ""

        query = f"""
        WITH RECURSIVE traversal(item, strength, depth, path) AS (
            SELECT node_id, 1.0::DOUBLE / COUNT(*) OVER (), 0, [node_id]
            FROM {self.table_name}
            WHERE node_id IN ({seeds_sql})
            UNION ALL
            SELECT e.target,
                   t.strength * e.score_val,
                   t.depth + 1,
                   list_append(t.path, e.target)
            FROM traversal t
            JOIN _decay_edges_scored e ON t.item = e.source
            WHERE t.depth < {max_depth}
              AND NOT list_contains(t.path, e.target)
        )
        SELECT
            item                                        AS recommended_item,
            SUM(strength)                               AS decay_score,
            MIN(depth)                                  AS min_hops,
            arg_max(array_to_string(path, ' -> '), strength) AS reason
        FROM traversal
        WHERE depth > 0 {exc}
        GROUP BY item
        ORDER BY decay_score DESC
        LIMIT {n}
        """
        return self.conn.query(query)

    def edge_table(self) -> pa.Table:
        """Return the full decayed edge table for inspection."""
        if not self._built:
            self.build()
        return self.conn.query("SELECT * FROM _decay_edges_scored ORDER BY score_val DESC")


# ---------------------------------------------------------------------------
# Algorithm 2 – Sliding Window Graph
# ---------------------------------------------------------------------------


class SlidingWindowGraph:
    """
    Rebuilds duwhal's graph using only interactions that fall inside a
    rolling temporal window ending at ``anchor_date``.

    This lets you ask: "what does the graph look like as of time T?"
    Sliding the anchor forward produces a sequence of graph snapshots.

    Parameters
    ----------
    dw : Duwhal
        A loaded Duwhal engine with sort_column data.
    window_days : int
        Width of the rolling window.
    anchor_date : str, optional
        End of the window (ISO date). Defaults to MAX(sort_column).
    """

    def __init__(
        self,
        dw: Any,  # duwhal.api.Duwhal – kept as Any to avoid circular import at runtime
        window_days: int = 30,
        anchor_date: Optional[str] = None,
    ):
        self.dw = dw
        self.window_days = window_days
        self.anchor_date = anchor_date

    def _window_bounds(self) -> tuple[str, str]:
        if self.anchor_date:
            anchor = f"TIMESTAMP '{self.anchor_date}'"
        else:
            anchor = f"(SELECT MAX(sort_column::TIMESTAMP) FROM {self.dw.table_name})"
        return anchor, f"{anchor} - INTERVAL {self.window_days} DAY"

    def _create_window_view(self) -> None:
        anchor, start = self._window_bounds()
        self.dw.conn.execute(f"""
            CREATE OR REPLACE TEMP VIEW _window_interactions AS
            SELECT set_id, node_id, sort_column
            FROM {self.dw.table_name}
            WHERE sort_column::TIMESTAMP BETWEEN {start} AND {anchor}
        """)

    def fit(self, **graph_kwargs: Any) -> SlidingWindowGraph:
        """Build graph on the windowed view."""
        self._create_window_view()
        original = self.dw.table_name
        self.dw.table_name = "_window_interactions"
        self.dw.fit_graph(**graph_kwargs)
        self.dw.table_name = original
        return self

    def recommend(self, seed_items: list[str], n: int = 10, **kwargs: Any) -> pa.Table:
        return self.dw.recommend_graph(seed_items, n=n, **kwargs)

    def slide(self, new_anchor: str) -> SlidingWindowGraph:
        """Move the window to a new anchor date and rebuild."""
        self.anchor_date = new_anchor
        return self.fit()


# ---------------------------------------------------------------------------
# Algorithm 3 – Temporal Snapshot Differ
# ---------------------------------------------------------------------------


@dataclass
class EdgeDelta:
    source: str
    target: str
    weight_before: float
    weight_after: float

    @property
    def delta(self) -> float:
        return self.weight_after - self.weight_before

    @property
    def kind(self) -> str:
        if self.weight_before == 0:
            return "emerging"
        if self.weight_after == 0:
            return "fading"
        return "strengthening" if self.delta > 0 else "weakening"


class TemporalSnapshotDiffer:
    """
    Splits the timeline into equal-width windows and builds one co-occurrence
    graph per window. Then diffs adjacent snapshots to expose:

    - **Emerging** edges  (new co-occurrences)
    - **Fading** edges    (disappeared co-occurrences)
    - **Strengthening**   (weight increased)
    - **Weakening**       (weight decreased)

    Parameters
    ----------
    dw : Duwhal
        A loaded Duwhal engine with sort_column.
    n_windows : int
        Number of equal-width temporal windows.
    min_cooccurrence : int
        Minimum raw count for an edge to be considered.
    """

    def __init__(
        self,
        dw: Any,
        n_windows: int = 4,
        min_cooccurrence: int = 1,
    ):
        self.dw = dw
        self.n_windows = n_windows
        self.min_cooccurrence = min_cooccurrence
        self.snapshots: list[dict[tuple[str, str], float]] = []
        self.window_labels: list[str] = []

    def _build_window_edges(self, start: str, end: str) -> dict[tuple[str, str], float]:
        result = self.dw.conn.query(f"""
            SELECT a.node_id AS source, b.node_id AS target, COUNT(*) AS w
            FROM {self.dw.table_name} a
            JOIN {self.dw.table_name} b
                ON  a.set_id  = b.set_id
                AND a.node_id != b.node_id
            WHERE a.sort_column::TIMESTAMP BETWEEN TIMESTAMP '{start}' AND TIMESTAMP '{end}'
            GROUP BY 1, 2
            HAVING w >= {self.min_cooccurrence}
        """).to_pylist()
        return {(r["source"], r["target"]): float(r["w"]) for r in result}

    def fit(self) -> TemporalSnapshotDiffer:
        bounds = self.dw.conn.execute(
            f"SELECT MIN(sort_column::TIMESTAMP), MAX(sort_column::TIMESTAMP) "
            f"FROM {self.dw.table_name}"
        ).fetchone()
        t_min, t_max = bounds

        total_seconds = (t_max - t_min).total_seconds()
        step = total_seconds / self.n_windows

        self.snapshots.clear()
        self.window_labels.clear()

        for i in range(self.n_windows):
            start = t_min + pd.Timedelta(seconds=i * step)
            end = t_min + pd.Timedelta(seconds=(i + 1) * step)
            label = f"W{i + 1}[{start.date()}\u2192{end.date()}]"
            self.window_labels.append(label)
            self.snapshots.append(self._build_window_edges(str(start), str(end)))

        return self

    def diff(self, window_a: int = 0, window_b: int = 1) -> list[EdgeDelta]:
        """Compare two snapshot indices and return a list of EdgeDeltas."""
        a, b = self.snapshots[window_a], self.snapshots[window_b]
        all_edges = set(a) | set(b)
        deltas = [
            EdgeDelta(
                source=src,
                target=tgt,
                weight_before=a.get((src, tgt), 0.0),
                weight_after=b.get((src, tgt), 0.0),
            )
            for src, tgt in all_edges
        ]
        return sorted(deltas, key=lambda d: abs(d.delta), reverse=True)

    def full_timeline(self) -> pd.DataFrame:
        """Return a long-format DataFrame of all edge weights over all windows."""
        rows = []
        for i, (snap, label) in enumerate(zip(self.snapshots, self.window_labels)):
            for (src, tgt), w in snap.items():
                rows.append(
                    {"window": i, "label": label, "source": src, "target": tgt, "weight": w}
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm 4 – Temporal SCC Tracker
# ---------------------------------------------------------------------------


@dataclass
class SCCSnapshot:
    window: int
    label: str
    sccs: dict[int, list[str]]  # scc_id → members
    sink_ids: set = field(default_factory=set)


class TemporalSCCTracker:
    """
    Runs duwhal's SinkSCCFinder (Tarjan's algorithm) on each temporal window
    and tracks community evolution:

    - **Stable**     SCCs that persist across consecutive windows
    - **Emerging**   SCCs that appear for the first time
    - **Dissolving** SCCs that disappear
    - **Merging**    Two SCCs merge into one larger SCC
    - **Splitting**  One SCC breaks into smaller ones

    Parameters
    ----------
    dw : Duwhal
        A loaded Duwhal engine with sort_column.
    n_windows : int
        Number of temporal windows.
    min_cooccurrence : int
        Minimum co-occurrence for an edge to be included.
    min_confidence : float
        Minimum transition probability for edge inclusion.
    """

    def __init__(
        self,
        dw: Any,
        n_windows: int = 4,
        min_cooccurrence: int = 2,
        min_confidence: float = 0.0,
    ):
        self.dw = dw
        self.n_windows = n_windows
        self.min_cooccurrence = min_cooccurrence
        self.min_confidence = min_confidence
        self.timeline: list[SCCSnapshot] = []

    def _snapshot_sccs(self, start: str, end: str) -> dict[int, list[str]]:
        self.dw.conn.execute(f"""
            CREATE OR REPLACE TEMP VIEW _scc_window AS
            SELECT set_id, node_id
            FROM {self.dw.table_name}
            WHERE sort_column::TIMESTAMP BETWEEN TIMESTAMP '{start}' AND TIMESTAMP '{end}'
        """)
        finder = SinkSCCFinder(
            self.dw.conn,
            table_name="_scc_window",
            min_cooccurrence=self.min_cooccurrence,
        )
        result = finder.find(min_confidence=self.min_confidence).to_pylist()
        sccs: dict[int, list[str]] = {}
        for row in result:
            sccs.setdefault(row["scc_id"], []).append(row["node"])
        return sccs

    def fit(self) -> TemporalSCCTracker:
        bounds = self.dw.conn.execute(
            f"SELECT MIN(sort_column::TIMESTAMP), MAX(sort_column::TIMESTAMP) "
            f"FROM {self.dw.table_name}"
        ).fetchone()
        t_min, t_max = bounds
        total_seconds = (t_max - t_min).total_seconds()
        step = total_seconds / self.n_windows

        self.timeline.clear()
        for i in range(self.n_windows):
            start = t_min + pd.Timedelta(seconds=i * step)
            end = t_min + pd.Timedelta(seconds=(i + 1) * step)
            label = f"W{i + 1}[{start.date()}\u2192{end.date()}]"
            sccs = self._snapshot_sccs(str(start), str(end))
            self.timeline.append(SCCSnapshot(window=i, label=label, sccs=sccs))

        return self

    def _jaccard(self, a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

    def evolution_report(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        For each consecutive window pair, classify each SCC as:
        stable / emerging / dissolving / merging / splitting.
        """
        rows = []
        for i in range(len(self.timeline) - 1):
            prev_snap = self.timeline[i]
            curr_snap = self.timeline[i + 1]
            prev_sccs = list(prev_snap.sccs.values())
            curr_sccs = list(curr_snap.sccs.values())

            matched_prev: set[int] = set()
            matched_curr: set[int] = set()

            for ci, curr in enumerate(curr_sccs):
                best_j, best_score = -1, 0.0
                for pi, prev in enumerate(prev_sccs):
                    score = self._jaccard(prev, curr)
                    if score > best_score:
                        best_score, best_j = score, pi

                if best_score >= threshold:
                    matched_prev.add(best_j)
                    matched_curr.add(ci)
                    rows.append(
                        {
                            "transition": f"{prev_snap.label} \u2192 {curr_snap.label}",
                            "event": "stable" if best_score > 0.8 else "evolving",
                            "prev_members": "|".join(sorted(prev_sccs[best_j])),
                            "curr_members": "|".join(sorted(curr)),
                            "jaccard": round(best_score, 3),
                        }
                    )

            for pi, prev in enumerate(prev_sccs):
                if pi not in matched_prev:
                    rows.append(
                        {
                            "transition": f"{prev_snap.label} \u2192 {curr_snap.label}",
                            "event": "dissolving",
                            "prev_members": "|".join(sorted(prev)),
                            "curr_members": "",
                            "jaccard": 0.0,
                        }
                    )

            for ci, curr in enumerate(curr_sccs):
                if ci not in matched_curr:
                    rows.append(
                        {
                            "transition": f"{prev_snap.label} \u2192 {curr_snap.label}",
                            "event": "emerging",
                            "prev_members": "",
                            "curr_members": "|".join(sorted(curr)),
                            "jaccard": 0.0,
                        }
                    )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm 5 – Directed Temporal Graph
# ---------------------------------------------------------------------------


class DirectedTemporalGraph:
    """
    The stock duwhal graph is *undirected* — co-occurrence is symmetric.
    This algorithm builds a *directed* graph from the temporal order of events,
    where A → B means A preceded B within the same session.

    Edge weight = probability that B follows A (transition probability).

    The directed topology is then used with duwhal's recursive CTE traversal
    to produce "what is likely to happen next" recommendations.

    Parameters
    ----------
    conn : DuckDBConnection
    table_name : str
        Must have set_id, node_id, and sort_column.
    max_gap : int, optional
        Maximum positional gap between A and B within a session.
        ``None`` means any ordering within a session is valid.
    min_support : int
        Minimum count of the A→B transition.
    """

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        max_gap: Optional[int] = None,
        min_support: int = 1,
    ):
        self.conn = conn
        self.table_name = table_name
        self.max_gap = max_gap
        self.min_support = min_support
        self._built = False

    def build(self) -> DirectedTemporalGraph:
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _dtg_ranked AS
            SELECT
                set_id,
                node_id,
                ROW_NUMBER() OVER (
                    PARTITION BY set_id
                    ORDER BY sort_column::TIMESTAMP
                ) AS pos
            FROM {self.table_name}
        """)

        gap_filter = (
            f"AND (b.pos - a.pos - 1) <= {self.max_gap}" if self.max_gap is not None else ""
        )

        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _dtg_raw AS
            SELECT
                a.node_id AS source,
                b.node_id AS target,
                COUNT(*) AS cnt
            FROM _dtg_ranked a
            JOIN _dtg_ranked b
                ON  a.set_id  = b.set_id
                AND b.pos     > a.pos
                {gap_filter}
            GROUP BY 1, 2
            HAVING cnt >= {self.min_support}
        """)

        self.conn.execute("""
            CREATE OR REPLACE TEMP TABLE _dtg_totals AS
            SELECT source, SUM(cnt) AS total_out
            FROM _dtg_raw
            GROUP BY source
        """)

        self.conn.execute("""
            CREATE OR REPLACE TEMP TABLE _dtg_edges AS
            SELECT
                r.source,
                r.target,
                r.cnt                             AS weight,
                r.cnt::DOUBLE / t.total_out       AS score_val
            FROM _dtg_raw r
            JOIN _dtg_totals t ON r.source = t.source
        """)

        self._built = True
        return self

    def recommend_next(
        self,
        seed_items: list[str],
        max_depth: int = 3,
        n: int = 10,
        exclude_seed: bool = True,
    ) -> pa.Table:
        """
        Follow directed edges from seed nodes (DFS with cumulative probability).
        Returns ranked predictions of what comes next.
        """
        if not self._built:
            self.build()

        seeds_sql = ", ".join(f"'{s}'" for s in seed_items)
        exc = f"AND item NOT IN ({seeds_sql})" if exclude_seed else ""

        query = f"""
        WITH RECURSIVE traversal(item, strength, depth, path) AS (
            SELECT node_id, 1.0::DOUBLE / COUNT(*) OVER (), 0, [node_id]
            FROM {self.table_name}
            WHERE node_id IN ({seeds_sql})
            UNION ALL
            SELECT e.target,
                   t.strength * e.score_val,
                   t.depth + 1,
                   list_append(t.path, e.target)
            FROM traversal t
            JOIN _dtg_edges e ON t.item = e.source
            WHERE t.depth < {max_depth}
              AND NOT list_contains(t.path, e.target)
        )
        SELECT
            item                                             AS next_item,
            SUM(strength)                                    AS transition_prob,
            MIN(depth)                                       AS steps,
            arg_max(array_to_string(path, ' -> '), strength) AS path
        FROM traversal
        WHERE depth > 0 {exc}
        GROUP BY item
        ORDER BY transition_prob DESC
        LIMIT {n}
        """
        return self.conn.query(query)

    def edge_table(self) -> pa.Table:
        """Inspect the directed edge table."""
        if not self._built:
            self.build()
        return self.conn.query("SELECT * FROM _dtg_edges ORDER BY score_val DESC")

    def reachable_from(self, node: str, max_depth: int = 3) -> pa.Table:
        """All nodes reachable from a given node following directed edges."""
        return self.recommend_next([node], max_depth=max_depth, n=1000, exclude_seed=True)
