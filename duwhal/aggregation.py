"""
duwhal.aggregation
==================
Analysis aggregation layer for time-variant graph-modelled data.

Builds on top of :mod:`duwhal.temporal` and duwhal's graph infrastructure.
All aggregators are DuckDB-first and return pandas DataFrames.

Five aggregators:

1. NodeTemporalAggregator      — Per-node metrics across windows
                                  (activity, degree, trend, churn, lifetime)
2. EdgeLifecycleAggregator     — Per-edge trajectory
                                  (stability, volatility, trend, peak window)
3. GraphSnapshotAggregator     — Whole-graph summary per window
                                  (density, new/churned nodes, SCC coverage)
4. CommunityAggregator         — SCC-level cohesion and bridge detection
5. NodeCorrelationMatrix       — Cross-node activity correlation
                                  (synchrony + lagged leading/lagging indicators)

Composite:

  TemporalAggregationReport    — Runs all five; surfaces a unified summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from duwhal.core.connection import DuckDBConnection
from duwhal.temporal import TemporalSCCTracker, TemporalSnapshotDiffer

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal helper — shared window SQL builder
# ---------------------------------------------------------------------------


class _WindowBuilder:
    """Equal-width temporal window CTE builder, shared across all aggregators."""

    def __init__(self, conn: DuckDBConnection, table_name: str, n_windows: int):
        self.conn = conn
        self.table_name = table_name
        self.n_windows = n_windows
        self._bounds: Optional[tuple] = None
        self._labels: Optional[list[str]] = None

    def _resolve_bounds(self) -> None:
        if self._bounds is None:
            self._bounds = self.conn.execute(
                f"SELECT MIN(sort_column::TIMESTAMP), MAX(sort_column::TIMESTAMP) "
                f"FROM {self.table_name}"
            ).fetchone()

    @property
    def t_min(self):
        self._resolve_bounds()
        return self._bounds[0]

    @property
    def t_max(self):
        self._resolve_bounds()
        return self._bounds[1]

    @property
    def step_seconds(self) -> float:
        return (self.t_max - self.t_min).total_seconds() / self.n_windows

    @property
    def window_labels(self) -> list[str]:
        if self._labels is None:
            self._labels = []
            for i in range(self.n_windows):
                s = self.t_min + pd.Timedelta(seconds=i * self.step_seconds)
                e = self.t_min + pd.Timedelta(seconds=(i + 1) * self.step_seconds)
                self._labels.append(f"W{i + 1}[{s.date()}\u2192{e.date()}]")
        return self._labels

    def cte_sql(self, alias: str = "wins") -> str:
        parts = []
        for i in range(self.n_windows):
            s = self.t_min + pd.Timedelta(seconds=i * self.step_seconds)
            e = self.t_min + pd.Timedelta(seconds=(i + 1) * self.step_seconds)
            parts.append(
                f"SELECT {i} AS win, "
                f"TIMESTAMP '{s}' AS wstart, "
                f"TIMESTAMP '{e}' AS wend, "
                f"'{self.window_labels[i]}' AS label"
            )
        return f"{alias} AS (\n  " + "\n  UNION ALL\n  ".join(parts) + "\n)"


# ---------------------------------------------------------------------------
# 1. NodeTemporalAggregator
# ---------------------------------------------------------------------------


class NodeTemporalAggregator:
    """
    Per-node metrics across every temporal window.

    ``per_window()`` columns
    ------------------------
    win, label, node_id, activity, sessions, out_degree

    ``rollup()`` columns
    --------------------
    node_id, first_window, last_window, peak_window,
    active_windows, lifetime_windows, total_activity,
    mean_activity, activity_std, activity_trend,
    total_sessions, status

    status values: ``'active'`` | ``'churned'`` | ``'emerging'`` | ``'absent'``

    ``activity_trend > 0`` → growing node, ``< 0`` → fading node.
    """

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        n_windows: int = 4,
        min_cooccurrence: int = 1,
    ):
        self.conn = conn
        self.table_name = table_name
        self.n_windows = n_windows
        self.min_cooccurrence = min_cooccurrence
        self._wb = _WindowBuilder(conn, table_name, n_windows)

    def per_window(self) -> pd.DataFrame:
        wins_cte = self._wb.cte_sql()
        df = self.conn.query(f"""
        WITH {wins_cte},
        activity AS (
            SELECT w.win, w.label, i.node_id,
                   COUNT(*)                  AS activity,
                   COUNT(DISTINCT i.set_id)  AS sessions
            FROM {self.table_name} i
            JOIN wins w ON i.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1, 2, 3
        ),
        out_deg AS (
            SELECT w.win, a.node_id,
                   COUNT(DISTINCT b.node_id) AS out_degree
            FROM {self.table_name} a
            JOIN {self.table_name} b
                ON  a.set_id  = b.set_id AND a.node_id != b.node_id
            JOIN wins w ON a.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1, 2
            HAVING COUNT(*) >= {self.min_cooccurrence}
        )
        SELECT a.*, COALESCE(d.out_degree, 0) AS out_degree
        FROM activity a LEFT JOIN out_deg d USING(win, node_id)
        ORDER BY a.node_id, a.win
        """).to_pandas()
        return df

    def rollup(self) -> pd.DataFrame:
        pw = self.per_window()
        n = self.n_windows
        labels = self._wb.window_labels

        all_nodes = pw["node_id"].unique()
        grid = pd.DataFrame(
            [(nd, w) for nd in all_nodes for w in range(n)],
            columns=["node_id", "win"],
        )
        full = grid.merge(
            pw[["node_id", "win", "activity", "sessions", "out_degree"]],
            on=["node_id", "win"],
            how="left",
        ).fillna(0)

        def _trend(arr: np.ndarray) -> float:
            x = np.arange(len(arr), dtype=float)
            y = arr.astype(float)
            denom = np.cov(x, y, ddof=1)[0, 0]
            return float(np.cov(x, y, ddof=1)[0, 1] / denom) if denom != 0 else 0.0

        rows = []
        for node, grp in full.groupby("node_id"):
            acts = grp.sort_values("win")["activity"].values.astype(float)
            sess = grp.sort_values("win")["sessions"].values.astype(float)
            mask = acts > 0
            active_wins = list(np.where(mask)[0])

            first_w = int(active_wins[0]) if active_wins else -1
            last_w = int(active_wins[-1]) if active_wins else -1
            peak_w = int(np.argmax(acts))

            if active_wins:
                status = "emerging" if first_w > 0 else ("churned" if last_w < n - 1 else "active")
            else:
                status = "absent"

            rows.append(
                {
                    "node_id": node,
                    "first_window": labels[first_w] if first_w >= 0 else None,
                    "last_window": labels[last_w] if last_w >= 0 else None,
                    "peak_window": labels[peak_w],
                    "active_windows": int(mask.sum()),
                    "lifetime_windows": int(last_w - first_w + 1) if first_w >= 0 else 0,
                    "total_activity": int(acts.sum()),
                    "mean_activity": round(float(acts[mask].mean()), 3) if mask.any() else 0.0,
                    "activity_std": round(float(acts.std(ddof=1)), 3),
                    "activity_trend": round(_trend(acts), 3),
                    "total_sessions": int(sess.sum()),
                    "status": status,
                }
            )

        return (
            pd.DataFrame(rows).sort_values("total_activity", ascending=False).reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 2. EdgeLifecycleAggregator
# ---------------------------------------------------------------------------


class EdgeLifecycleAggregator:
    """
    Per-edge (source, target) lifecycle metrics across windows.

    ``per_window()`` — long-format weight per window (from TemporalSnapshotDiffer)

    ``rollup()`` columns
    --------------------
    source, target, active_windows, stability,
    weight_mean, weight_std, weight_max, weight_trend,
    peak_window, first_window, last_window,
    edge_kind  (``'persistent'`` | ``'transient'`` | ``'sporadic'``)

    stability = active_windows / n_windows

    weight_trend > 0 → strengthening edge.
    """

    def __init__(
        self,
        dw: Any,
        n_windows: int = 4,
        min_cooccurrence: int = 1,
    ):
        self.dw = dw
        self.n_windows = n_windows
        self._differ = TemporalSnapshotDiffer(
            dw, n_windows=n_windows, min_cooccurrence=min_cooccurrence
        )
        self._wb = _WindowBuilder(dw.conn, dw.table_name, n_windows)
        self._fitted = False

    def _ensure_fit(self) -> None:
        if not self._fitted:
            self._differ.fit()
            self._fitted = True

    def per_window(self) -> pd.DataFrame:
        self._ensure_fit()
        return self._differ.full_timeline()

    def rollup(self) -> pd.DataFrame:
        self._ensure_fit()
        tl = self._differ.full_timeline()
        labels = self._wb.window_labels
        n = self.n_windows

        all_edges = tl[["source", "target"]].drop_duplicates()
        all_wins = pd.DataFrame({"window": range(n), "label": labels})
        full = (
            all_edges.merge(all_wins, how="cross")
            .merge(
                tl[["source", "target", "window", "weight"]],
                on=["source", "target", "window"],
                how="left",
            )
            .fillna(0)
        )

        rows = []
        for (src, tgt), grp in full.groupby(["source", "target"]):
            grp = grp.sort_values("window")
            weights = grp["weight"].values.astype(float)
            active = weights > 0
            active_w = int(active.sum())

            x = np.arange(n, dtype=float)
            trend = float(np.polyfit(x, weights, 1)[0]) if active_w > 1 else 0.0
            peak = int(np.argmax(weights))
            first = int(np.argmax(active)) if active.any() else -1
            last = int(n - 1 - np.argmax(active[::-1])) if active.any() else -1

            kind = (
                "persistent" if active_w > n * 0.5 else "transient" if active_w == 1 else "sporadic"
            )

            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "active_windows": active_w,
                    "stability": round(active_w / n, 3),
                    "weight_mean": round(float(weights[active].mean()), 3) if active.any() else 0.0,
                    "weight_std": round(float(weights.std(ddof=1)), 3),
                    "weight_max": round(float(weights.max()), 3),
                    "weight_trend": round(trend, 4),
                    "peak_window": labels[peak],
                    "first_window": labels[first] if first >= 0 else None,
                    "last_window": labels[last] if last >= 0 else None,
                    "edge_kind": kind,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(["stability", "weight_mean"], ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 3. GraphSnapshotAggregator
# ---------------------------------------------------------------------------


class GraphSnapshotAggregator:
    """
    Whole-graph structural metrics per temporal window.

    Output columns
    --------------
    win, label, n_nodes, n_edges, density, avg_degree,
    n_sessions, new_nodes, churned_nodes,
    n_sccs, largest_scc_size, scc_coverage
    """

    def __init__(
        self,
        dw: Any,
        n_windows: int = 4,
        min_cooccurrence: int = 1,
        scc_tracker: Optional[TemporalSCCTracker] = None,
    ):
        self.dw = dw
        self.n_windows = n_windows
        self.min_cooccurrence = min_cooccurrence
        self._wb = _WindowBuilder(dw.conn, dw.table_name, n_windows)
        self._scc_tracker = scc_tracker

    def compute(self) -> pd.DataFrame:
        wins_cte = self._wb.cte_sql()
        labels = self._wb.window_labels
        n = self.n_windows

        structural = self.dw.conn.query(f"""
        WITH {wins_cte},
        node_win AS (
            SELECT w.win, i.node_id
            FROM {self.dw.table_name} i
            JOIN wins w ON i.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1, 2
        ),
        edge_win AS (
            SELECT w.win,
                   LEAST(a.node_id, b.node_id)    AS src,
                   GREATEST(a.node_id, b.node_id) AS tgt,
                   COUNT(*) AS cooc
            FROM {self.dw.table_name} a
            JOIN {self.dw.table_name} b
                ON  a.set_id = b.set_id AND a.node_id < b.node_id
            JOIN wins w ON a.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1, 2, 3
            HAVING cooc >= {self.min_cooccurrence}
        ),
        session_win AS (
            SELECT w.win, COUNT(DISTINCT i.set_id) AS n_sessions
            FROM {self.dw.table_name} i
            JOIN wins w ON i.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1
        ),
        counts AS (
            SELECT nw.win,
                   COUNT(DISTINCT nw.node_id)              AS n_nodes,
                   COUNT(DISTINCT ew.src || '|' || ew.tgt) AS n_edges
            FROM node_win nw LEFT JOIN edge_win ew USING(win)
            GROUP BY 1
        )
        SELECT c.win, c.n_nodes, c.n_edges,
               ROUND(c.n_edges::DOUBLE / NULLIF(c.n_nodes*(c.n_nodes-1)/2.0, 0), 4) AS density,
               ROUND(2.0 * c.n_edges / NULLIF(c.n_nodes, 0), 3)                      AS avg_degree,
               COALESCE(s.n_sessions, 0) AS n_sessions
        FROM counts c LEFT JOIN session_win s USING(win)
        ORDER BY c.win
        """).to_pandas()
        structural["label"] = [labels[i] for i in structural["win"]]

        churn = self.dw.conn.query(f"""
        WITH {wins_cte},
        node_win AS (
            SELECT w.win, i.node_id
            FROM {self.dw.table_name} i
            JOIN wins w ON i.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
            GROUP BY 1, 2
        ),
        fl AS (
            SELECT node_id, MIN(win) AS first_win, MAX(win) AS last_win
            FROM node_win GROUP BY node_id
        ),
        all_wins AS (
            SELECT DISTINCT win FROM node_win
        ),
        new_per_win AS (
            SELECT nw.win, COUNT(*) AS new_nodes
            FROM node_win nw
            JOIN fl ON fl.node_id = nw.node_id AND fl.first_win = nw.win
            GROUP BY nw.win
        ),
        churn_per_win AS (
            SELECT fl.last_win + 1 AS win, COUNT(*) AS churned_nodes
            FROM fl
            WHERE fl.last_win < {n - 1}
            GROUP BY fl.last_win + 1
        )
        SELECT aw.win,
               COALESCE(np.new_nodes,     0) AS new_nodes,
               COALESCE(cp.churned_nodes, 0) AS churned_nodes
        FROM all_wins aw
        LEFT JOIN new_per_win   np USING(win)
        LEFT JOIN churn_per_win cp USING(win)
        ORDER BY aw.win
        """).to_pandas()

        structural = structural.merge(churn, on="win", how="left").fillna(0)

        if self._scc_tracker is not None:
            scc_rows = []
            for snap in self._scc_tracker.timeline:
                all_members = [m for v in snap.sccs.values() for m in v]
                scc_sizes = [len(v) for v in snap.sccs.values()]
                scc_rows.append(
                    {
                        "win": snap.window,
                        "n_sccs": len(snap.sccs),
                        "largest_scc_size": max(scc_sizes) if scc_sizes else 0,
                        "scc_node_count": len(all_members),
                    }
                )
            scc_df = pd.DataFrame(scc_rows)
            structural = structural.merge(scc_df, on="win", how="left").fillna(0)
            structural["scc_coverage"] = (
                structural["scc_node_count"] / structural["n_nodes"].replace(0, float("nan"))
            ).round(3)
        else:
            structural["n_sccs"] = None
            structural["largest_scc_size"] = None
            structural["scc_coverage"] = None

        cols = [
            "win",
            "label",
            "n_nodes",
            "n_edges",
            "density",
            "avg_degree",
            "n_sessions",
            "new_nodes",
            "churned_nodes",
            "n_sccs",
            "largest_scc_size",
            "scc_coverage",
        ]
        return structural[[c for c in cols if c in structural.columns]]


# ---------------------------------------------------------------------------
# 4. CommunityAggregator
# ---------------------------------------------------------------------------


class CommunityAggregator:
    """
    SCC-level cohesion and bridge node detection per window.

    Output columns
    --------------
    window, window_label, scc_id, size, members,
    internal_edges, possible_edges, cohesion,
    external_edges, bridge_nodes, n_bridge_nodes

    cohesion     = internal_edges / possible_edges  (1 = fully clique-like)

    bridge_nodes = SCC members that also co-occur with non-SCC nodes
    """

    def __init__(self, dw: Any, scc_tracker: TemporalSCCTracker):
        self.dw = dw
        self.tracker = scc_tracker

    def _window_edges(self, win_idx: int) -> dict[tuple[str, str], int]:
        wb = _WindowBuilder(self.dw.conn, self.dw.table_name, self.tracker.n_windows)
        step = (wb.t_max - wb.t_min).total_seconds() / self.tracker.n_windows
        wstart = wb.t_min + pd.Timedelta(seconds=win_idx * step)
        wend = wb.t_min + pd.Timedelta(seconds=(win_idx + 1) * step)
        rows = self.dw.conn.query(f"""
            SELECT LEAST(a.node_id, b.node_id)    AS src,
                   GREATEST(a.node_id, b.node_id) AS tgt,
                   COUNT(*) AS cooc
            FROM {self.dw.table_name} a
            JOIN {self.dw.table_name} b ON a.set_id = b.set_id AND a.node_id < b.node_id
            WHERE a.sort_column::TIMESTAMP BETWEEN TIMESTAMP '{wstart}' AND TIMESTAMP '{wend}'
            GROUP BY 1, 2
        """).to_pylist()
        return {(r["src"], r["tgt"]): r["cooc"] for r in rows}

    def compute(self) -> pd.DataFrame:
        rows = []
        for snap in self.tracker.timeline:
            edges = self._window_edges(snap.window)
            for scc_id, members in snap.sccs.items():
                members_set = set(members)
                possible = len(members) * (len(members) - 1) / 2
                internal, external = 0, 0
                bridge_nodes: set[str] = set()
                for (src, tgt), w in edges.items():
                    src_in = src in members_set
                    tgt_in = tgt in members_set
                    if src_in and tgt_in:
                        internal += w
                    elif src_in or tgt_in:
                        external += w
                        bridge_nodes.add(src if src_in else tgt)
                rows.append(
                    {
                        "window": snap.window,
                        "window_label": snap.label,
                        "scc_id": scc_id,
                        "size": len(members),
                        "members": "|".join(sorted(members)),
                        "internal_edges": internal,
                        "possible_edges": int(possible),
                        "cohesion": round(internal / possible, 3) if possible > 0 else 1.0,
                        "external_edges": external,
                        "bridge_nodes": "|".join(sorted(bridge_nodes)),
                        "n_bridge_nodes": len(bridge_nodes),
                    }
                )
        return pd.DataFrame(rows).sort_values(["window", "scc_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. NodeCorrelationMatrix
# ---------------------------------------------------------------------------


class NodeCorrelationMatrix:
    """
    Cross-node activity correlation across temporal windows.

    ``synchrony()`` — Pearson correlation matrix (nodes × nodes).
      + → nodes rise and fall together (co-dependent)
      − → nodes are substitutes / anti-correlated

    ``leading_lagging()`` — Lagged cross-correlation for all pairs at lag 1..max_lag.
      Output: node_a, node_b, lag, correlation, relationship

      "node_a leads node_b" means node_a's activity at t predicts node_b at t+lag.
    """

    def __init__(self, dw: Any, n_windows: int = 4, max_lag: int = 2):
        self.dw = dw
        self.n_windows = n_windows
        self.max_lag = max_lag
        self._wb = _WindowBuilder(dw.conn, dw.table_name, n_windows)
        self._mat: Optional[pd.DataFrame] = None

    def _activity_matrix(self) -> pd.DataFrame:
        if self._mat is not None:
            return self._mat
        wins_cte = self._wb.cte_sql()
        long = self.dw.conn.query(f"""
        WITH {wins_cte}
        SELECT w.win, i.node_id, COUNT(*) AS activity
        FROM {self.dw.table_name} i
        JOIN wins w ON i.sort_column::TIMESTAMP BETWEEN w.wstart AND w.wend
        GROUP BY 1, 2 ORDER BY 1, 2
        """).to_pandas()
        self._mat = long.pivot_table(
            index="win", columns="node_id", values="activity", fill_value=0
        )
        return self._mat

    def synchrony(self) -> pd.DataFrame:
        """Symmetric Pearson correlation matrix."""
        mat = self._activity_matrix()
        corr = mat.corr(method="pearson").round(3)
        corr.index.name = corr.columns.name = "node"
        return corr

    def leading_lagging(self) -> pd.DataFrame:
        """Lagged cross-correlation for all node pairs.

        For each unordered pair (a, b) and each lag, only the direction with
        the strictly higher |correlation| is kept.  When both directions are
        equally strong the pair is omitted — reporting "A leads B" AND
        "B leads A" at the same lag would be contradictory.
        """
        mat = self._activity_matrix()
        nodes = mat.columns.tolist()
        rows = []
        for i, na in enumerate(nodes):
            for nb in nodes[i + 1 :]:
                a = mat[na].values.astype(float)
                b = mat[nb].values.astype(float)
                for lag in range(1, self.max_lag + 1):
                    if len(a) <= lag:
                        continue
                    la, fo_ab = a[:-lag], b[lag:]
                    lb, fo_ba = b[:-lag], a[lag:]
                    r_ab = (
                        float(np.corrcoef(la, fo_ab)[0, 1])
                        if la.std() > 0 and fo_ab.std() > 0
                        else 0.0
                    )
                    r_ba = (
                        float(np.corrcoef(lb, fo_ba)[0, 1])
                        if lb.std() > 0 and fo_ba.std() > 0
                        else 0.0
                    )
                    # Report only the dominant direction; skip symmetric ties
                    # to prevent the contradictory "A leads B" + "B leads A".
                    if abs(r_ab) > abs(r_ba) + 1e-6:
                        rows.append(
                            {
                                "node_a": na,
                                "node_b": nb,
                                "lag": lag,
                                "correlation": round(r_ab, 3),
                                "relationship": f"{na} leads {nb}",
                            }
                        )
                    elif abs(r_ba) > abs(r_ab) + 1e-6:
                        rows.append(
                            {
                                "node_a": nb,
                                "node_b": na,
                                "lag": lag,
                                "correlation": round(r_ba, 3),
                                "relationship": f"{nb} leads {na}",
                            }
                        )
                    # else: symmetric — skip to avoid contradictions
        df = pd.DataFrame(rows)
        return df.reindex(df["correlation"].abs().sort_values(ascending=False).index).reset_index(
            drop=True
        )


# ---------------------------------------------------------------------------
# Composite — TemporalAggregationReport
# ---------------------------------------------------------------------------


@dataclass
class TemporalAggregationReport:
    """
    Runs all five aggregators in one call.

    Attributes
    ----------
    node_per_window   : per-node per-window activity & degree
    node_rollup       : lifetime node stats (trend, status, churn)
    edge_per_window   : long-format edge weights per window
    edge_lifecycle    : per-edge stability, volatility, trend, kind
    graph_snapshots   : per-window whole-graph structural metrics
    community_metrics : SCC cohesion and bridge nodes per window
    synchrony_matrix  : symmetric node correlation matrix
    leading_lagging   : lagged correlation pairs (leading/lagging detection)
    window_labels     : human-readable window label list
    """

    node_per_window: pd.DataFrame
    node_rollup: pd.DataFrame
    edge_per_window: pd.DataFrame
    edge_lifecycle: pd.DataFrame
    graph_snapshots: pd.DataFrame
    community_metrics: pd.DataFrame
    synchrony_matrix: pd.DataFrame
    leading_lagging: pd.DataFrame
    window_labels: list[str]

    @classmethod
    def build(
        cls,
        dw: Any,
        n_windows: int = 4,
        min_cooccurrence: int = 1,
        max_lag: int = 2,
        scc_min_cooccurrence: int = 1,
    ) -> TemporalAggregationReport:
        tracker = TemporalSCCTracker(dw, n_windows=n_windows, min_cooccurrence=scc_min_cooccurrence)
        tracker.fit()

        node_agg = NodeTemporalAggregator(dw.conn, dw.table_name, n_windows, min_cooccurrence)
        edge_agg = EdgeLifecycleAggregator(dw, n_windows, min_cooccurrence)
        graph_agg = GraphSnapshotAggregator(dw, n_windows, min_cooccurrence, scc_tracker=tracker)
        comm_agg = CommunityAggregator(dw, tracker)
        corr_agg = NodeCorrelationMatrix(dw, n_windows, max_lag)

        return cls(
            node_per_window=node_agg.per_window(),
            node_rollup=node_agg.rollup(),
            edge_per_window=edge_agg.per_window(),
            edge_lifecycle=edge_agg.rollup(),
            graph_snapshots=graph_agg.compute(),
            community_metrics=comm_agg.compute(),
            synchrony_matrix=corr_agg.synchrony(),
            leading_lagging=corr_agg.leading_lagging(),
            window_labels=_WindowBuilder(dw.conn, dw.table_name, n_windows).window_labels,
        )

    def summary(self) -> None:
        """Print a narrative summary of the most salient findings."""
        sep = "=" * 65
        print(sep)
        print("TEMPORAL AGGREGATION REPORT")
        print(sep)

        print("\n-- Graph Snapshot Evolution --")
        print(
            self.graph_snapshots[
                [
                    "label",
                    "n_nodes",
                    "n_edges",
                    "density",
                    "n_sessions",
                    "new_nodes",
                    "churned_nodes",
                    "n_sccs",
                    "scc_coverage",
                ]
            ].to_string(index=False)
        )

        nr = self.node_rollup
        print("\n-- Node Lifecycle Highlights --")
        rising = nr[nr["activity_trend"] > 0].sort_values("activity_trend", ascending=False)
        falling = nr[nr["activity_trend"] < 0].sort_values("activity_trend")
        churned = nr[nr["status"] == "churned"]
        emerging = nr[nr["status"] == "emerging"]
        print(f"  Rising   : {rising['node_id'].tolist()}")
        print(f"  Falling  : {falling['node_id'].tolist()}")
        print(f"  Churned  : {churned['node_id'].tolist()}")
        print(f"  Emerging : {emerging['node_id'].tolist()}")

        el = self.edge_lifecycle
        persistent = el[el["edge_kind"] == "persistent"]
        transient = el[el["edge_kind"] == "transient"]
        print("\n-- Edge Lifecycle --")
        print(
            f"  Persistent ({len(persistent)}): "
            + ", ".join(f"{r.source}->{r.target}" for _, r in persistent.iterrows())
        )
        transient_sample = transient.head(5)
        suffix = f" ... (+{len(transient) - 5} more)" if len(transient) > 5 else ""
        print(
            f"  Transient  ({len(transient)}): "
            + ", ".join(f"{r.source}->{r.target}" for _, r in transient_sample.iterrows())
            + suffix
        )

        cm = self.community_metrics
        if not cm.empty:
            print("\n-- Community Cohesion per Window --")
            print(
                cm[["window_label", "scc_id", "size", "cohesion", "n_bridge_nodes"]].to_string(
                    index=False
                )
            )

        ll = self.leading_lagging
        if not ll.empty:
            print("\n-- Top Leading/Lagging Node Relationships --")
            print(
                ll.head(6)[["node_a", "node_b", "lag", "correlation", "relationship"]].to_string(
                    index=False
                )
            )

        print("\n" + sep)
