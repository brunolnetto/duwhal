from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd

    from .profiling.item_profiler import ItemProfiler
    from .profiling.rule_profiler import RuleProfiler

from duwhal.core.connection import DuckDBConnection
from duwhal.core.facets import merge_recommendation_tables, split_by_facet
from duwhal.core.ingestion import load_interaction_matrix, load_interactions
from duwhal.mining.association_rules import AssociationRules
from duwhal.mining.frequent_itemsets import FrequentItemsets
from duwhal.mining.sequences import SequentialPatterns
from duwhal.recommenders.graph import GraphRecommender
from duwhal.recommenders.item_cf import ItemCF
from duwhal.recommenders.popularity import PopularityRecommender


class Duwhal:
    """Unified engine for discovery and recommendations."""

    def __init__(
        self,
        database: Union[str, Path] = ":memory:",
        memory_limit: Optional[str] = None,
        threads: Optional[int] = None,
    ) -> None:
        self.conn = DuckDBConnection(database=database, memory_limit=memory_limit, threads=threads)
        self.table_name = "interactions"
        self._rules, self._cf_model, self._graph_model, self._pop_model = None, None, None, None

    @property
    def _graph(self) -> Optional[GraphRecommender]: return self._graph_model
    @property
    def _rules_cached(self) -> Optional[pa.Table]: return self._rules

    def load(self, data: Any, **kwargs) -> Duwhal:
        load_interactions(self.conn, data, table_name=self.table_name, **kwargs)
        return self

    def load_interactions(
        self,
        *args,
        facet_cols: Optional[List[str]] = None,
        facet_mode: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Load interactions with optional facet enrichment.

        Parameters
        ----------
        facet_cols:
            Columns in the source DataFrame that carry facet signals
            (e.g. ``["region", "day_period"]``).
        facet_mode:
            ``"composite"`` – build a composite context key (pattern 1).
            ``"entity"``    – inject facet pseudo-entities (pattern 3).
            ``None``        – no transformation (default).
        """
        return load_interactions(
            self.conn,
            *args,
            table_name=self.table_name,
            facet_cols=facet_cols,
            facet_mode=facet_mode,
            **kwargs,
        )

    def load_interaction_matrix(self, df: Any, append: bool = False) -> int:
        return load_interaction_matrix(self.conn, df, append=append, table_name=self.table_name)

    def frequent_itemsets(self, **kwargs) -> pa.Table:
        return FrequentItemsets(self.conn, table_name=self.table_name, **kwargs).fit()

    def association_rules(self, **kwargs) -> pa.Table:
        self._rules = AssociationRules(self.conn, table_name=self.table_name, **kwargs).fit()
        return self._rules

    def sequential_patterns(self, **kwargs) -> pa.Table:
        return SequentialPatterns(self.conn, table_name=self.table_name, **kwargs).fit()

    def fit_cf(self, **kwargs) -> Duwhal:
        self._cf_model = ItemCF(self.conn, table_name=self.table_name, **kwargs).fit()
        return self

    def recommend_cf(self, *args, **kwargs) -> pa.Table:
        if not self._cf_model: raise RuntimeError("Call fit_cf() first.")
        return self._cf_model.recommend(*args, **kwargs)

    def fit_graph(self, alpha: float = 0.0, **kwargs) -> Duwhal:
        self._graph_model = GraphRecommender(self.conn, table_name=self.table_name, alpha=alpha, **kwargs).build()
        return self

    def recommend_graph(self, *args, **kwargs) -> pa.Table:
        if not self._graph_model: self.fit_graph()
        return self._graph_model.recommend(*args, **kwargs)

    def score_basket(self, items: List[str]) -> float:
        if not self._graph_model: self.fit_graph()
        return self._graph_model.score_basket(items)

    def fit_popularity(self, strategy: str = "global", window_days: int = 30, **kwargs) -> Duwhal:
        self._pop_model = PopularityRecommender(self.conn, table_name=self.table_name, strategy=strategy, window_days=window_days, **kwargs).fit()
        return self

    def recommend_popular(self, *args, **kwargs) -> pa.Table:
        if not self._pop_model: self.fit_popularity()
        return self._pop_model.recommend(*args, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers for recommend_by_facet
    # ------------------------------------------------------------------

    def _run_on_df_slice(
        self,
        slice_df: "pd.DataFrame",
        set_col: str,
        node_col: str,
        seed_items: List[str],
        strategy: str,
        n: int,
        sort_col: Optional[str] = None,
        sort_callback: Optional[Callable] = None,
        **kw,
    ) -> pa.Table:
        """Fit + recommend on a pre-filtered DataFrame slice."""
        _empty = pa.table({
            "recommended_item": pa.array([], pa.string()),
            "total_strength":   pa.array([], pa.float64()),
        })
        if slice_df.empty:
            return _empty
        with Duwhal(memory_limit=self.conn._memory_limit, threads=self.conn._threads) as _db:
            _db.load_interactions(slice_df, set_col=set_col, node_col=node_col,
                                  sort_col=sort_col, sort_callback=sort_callback)
            try:
                result = _db.recommend(seed_items, strategy=strategy, n=n, **kw)
                return result if len(result) > 0 else _empty
            except Exception:
                return _empty

    def _coarsen_fallback(
        self,
        seed_items: List[str],
        df: "pd.DataFrame",
        set_col: str,
        node_col: str,
        slice_key: Dict[str, Any],
        strategy: str,
        n: int,
        fallback_merge: str,
        sort_col: Optional[str] = None,
        sort_callback: Optional[Callable] = None,
        **kw,
    ) -> pa.Table:
        """BFS through the facet lattice for a non-empty recommendation.

        Starting from *slice_key* (a dict mapping facet column → value), the
        method iteratively tries every way of dropping **one** facet at a time,
        collecting any coarser slice that yields a non-empty result.  Multiple
        non-empty candidates at the same BFS level are merged by
        *fallback_merge*.  If a full BFS level is still empty, the next level
        (two facets dropped) is tried, continuing until the global model
        (zero remaining facets).

        The strategy mirrors how a human would escalate: if *EU/morning* is
        empty, look at all *EU* and all *morning* slices independently.  If
        both are non-empty, blend them.  If neither helps, use the global model.
        """
        import itertools

        import pandas as _pd

        _empty = pa.table({
            "recommended_item": pa.array([], pa.string()),
            "total_strength":   pa.array([], pa.float64()),
        })
        facets = list(slice_key.keys())

        for n_drop in range(1, len(facets) + 1):
            candidates: List[pa.Table] = []
            for keep in itertools.combinations(facets, len(facets) - n_drop):
                coarser_key = {f: slice_key[f] for f in keep}
                keep_cols = [set_col, node_col] + ([sort_col] if sort_col else [])
                if coarser_key:
                    mask = _pd.Series([True] * len(df), index=df.index)
                    for col, val in coarser_key.items():
                        mask &= df[col].astype(str) == str(val)
                    coarser_slice = df[mask][keep_cols].drop_duplicates()
                else:
                    coarser_slice = df[keep_cols].drop_duplicates()  # global
                result = self._run_on_df_slice(
                    coarser_slice, set_col, node_col, seed_items, strategy, n,
                    sort_col=sort_col, sort_callback=sort_callback, **kw
                )
                if len(result) > 0:
                    candidates.append(result)

            if candidates:
                if len(candidates) == 1:
                    return candidates[0]
                return merge_recommendation_tables(
                    candidates, strategy=fallback_merge, n=n
                )

        return _empty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_by_facet(
        self,
        seed_items: List[str],
        df,
        set_col: str,
        node_col: str,
        facet_cols: Union[str, List[str]],
        *,
        strategy: str = "graph",
        n: int = 10,
        extra_cols: Optional[List[str]] = None,
        sort_col: Optional[str] = None,
        sort_callback: Optional[Callable] = None,
        fallback_merge: str = "union",
        **recommend_kwargs,
    ) -> Dict[str, pa.Table]:
        """Pattern 2 – parallel per-facet runs with granularity-coarsening fallback.

        Runs the chosen strategy on every facet slice.  When a slice yields
        no results (seed absent at the requested granularity), the method
        automatically moves to *granularity+1* by trying every possible
        one-facet relaxation via BFS through the facet lattice.

        If multiple coarser slices at the same BFS level each return non-empty
        results, they are blended by *fallback_merge*.

        Parameters
        ----------
        seed_items:
            Items to seed all models.
        df:
            Raw interaction DataFrame.
        set_col:
            Primary-context column.
        node_col:
            Item / entity column.
        facet_cols:
            Single column name **or** ordered list of column names.  Each
            unique value-combination becomes its own finest-grain slice.
            The order determines the lattice: leftmost facets are dropped
            last, so they act as the "most important" dimension.
        strategy:
            Recommendation strategy passed to ``recommend()``.
        n:
            Maximum recommendations per slice.
        extra_cols:
            Additional columns kept in each slice (e.g. ``["timestamp"]``).
        fallback_merge:
            How to blend when multiple coarser slices are all non-empty:
            ``"union"`` (default) – sum scores and re-rank.
            ``"best"``            – keep the highest total-score table.
            ``"first"``           – keep the first non-empty table.
        **recommend_kwargs:
            Forwarded to ``recommend()`` for every sub-model.

        Returns
        -------
        dict[str, pa.Table]
            ``{"global": …, "<slice_label>": …, …}`` where each slice label
            encodes all active facet dimensions, e.g.
            ``"region=EU|day_period=morning"``.
        """

        if isinstance(facet_cols, str):
            facet_cols = [facet_cols]

        # Ensure sort_col is in extra_cols so it's preserved in slices
        effective_extra = list(extra_cols or [])
        if sort_col and sort_col not in effective_extra:
            effective_extra.append(sort_col)

        slices = split_by_facet(df, set_col, node_col, facet_cols, extra_cols=effective_extra)
        results: Dict[str, pa.Table] = {}

        # Global model (always computed; serves as ultimate fallback)
        keep_global = [set_col, node_col] + ([sort_col] if sort_col else [])
        global_df = df[keep_global].drop_duplicates()
        results["global"] = self._run_on_df_slice(
            global_df, set_col, node_col, seed_items, strategy, n,
            sort_col=sort_col, sort_callback=sort_callback, **recommend_kwargs
        )

        # Per-slice models with automatic coarsening fallback
        for label, sub_df in slices.items():
            # Parse the label back into a {facet_col: value} dict
            slice_key = dict(
                part.split("=", 1)
                for part in label.split("|")
                if "=" in part
            )
            result = self._run_on_df_slice(
                sub_df, set_col, node_col, seed_items, strategy, n,
                sort_col=sort_col, sort_callback=sort_callback, **recommend_kwargs
            )
            if len(result) == 0:
                result = self._coarsen_fallback(
                    seed_items, df, set_col, node_col,
                    slice_key, strategy, n, fallback_merge,
                    sort_col=sort_col, sort_callback=sort_callback, **recommend_kwargs,
                )
            results[label] = result

        return results

    def _get_rule_matches(self, seeds: set, metric: str) -> List[Dict]:
        matches = []
        for row in self._rules.to_pylist():
            ants = set(row["antecedents"].split("|"))
            if ants.issubset(seeds) and row["consequents"] not in seeds:
                matches.append({"item_id": row["consequents"], "score": row[metric], "rule": f"{row['antecedents']} -> {row['consequents']}"})
        return matches

    def _filter_matches(self, matches: List[Dict], n: int) -> List[Dict]:
        matches.sort(key=lambda x: x["score"], reverse=True)
        unique, seen = [], set()
        for m in matches:
            if m["item_id"] not in seen:
                seen.add(m["item_id"]); unique.append(m)
                if len(unique) >= n: break
        return unique

    def recommend_by_rules(self, seed_items: List[str], n: int = 10, metric: str = "lift") -> pa.Table:
        if self._rules is None: raise RuntimeError("Call association_rules() first.")
        if metric not in AssociationRules.METRICS: raise ValueError(f"metric must be one of {AssociationRules.METRICS}")

        matches = self._get_rule_matches(set(seed_items), metric)
        if not matches:
            return pa.Table.from_batches([], schema=pa.schema([("item_id", pa.string()), ("score", pa.float64()), ("rule", pa.string())]))
        return pa.Table.from_pylist(self._filter_matches(matches, n))

    def _dispatch_recommendation(self, strategy: str, seed_items: List[str], n: int, kwargs: dict) -> pa.Table:
        if strategy == "graph": return self.recommend_graph(seed_items, n=n, **kwargs)
        if strategy == "cf": return self.recommend_cf(seed_items, n=n, **kwargs)
        if strategy == "rules": return self.recommend_by_rules(seed_items, n=n, **kwargs)
        if strategy.startswith("popular"): return self.recommend_popular(n=n, **kwargs)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _resolve_strategy(self, strategy: str) -> str:
        if strategy != "auto": return strategy
        if self._rules is not None: return "rules"
        if self._cf_model: return "cf"
        return "graph"

    def recommend(self, seed_items: Optional[List[str]] = None, strategy: str = "auto", n: int = 10, **kwargs) -> pa.Table:
        res = self._dispatch_recommendation(self._resolve_strategy(strategy), seed_items or [], n, kwargs)
        if "item_id" in res.column_names:
            res = res.rename_columns(["recommended_item" if c == "item_id" else c for c in res.column_names])
        return res

    def find_sink_sccs(self, min_cooccurrence: int = 5, min_confidence: float = 0.0) -> pa.Table:
        """Identifies Sink Strongly Connected Components (Equilibrium Communities)."""
        from .mining.sink_sccs import SinkSCCFinder
        return SinkSCCFinder(self.conn, table_name=self.table_name, min_cooccurrence=min_cooccurrence).find(min_confidence=min_confidence)

    def profile_rules(
        self,
        n_clusters: Union[int, str] = "auto",
        max_k: int = 8,
        random_state: int = 0,
    ) -> "RuleProfiler":
        """
        Cluster association rules in metric-space to reveal latent recommendation personas.

        Requires ``association_rules()`` to have been called first.

        Parameters
        ----------
        n_clusters : int or "auto"
            Number of clusters. ``"auto"`` selects the best k via silhouette score.
        max_k : int
            Upper bound for k search when ``n_clusters="auto"`` (default 8).
        random_state : int
            Reproducibility seed (default 0).

        Returns
        -------
        RuleProfiler
            A fitted profiler. Inspect ``.clusters`` or call ``.recommend()``.
        """
        from .profiling.rule_profiler import RuleProfiler
        if self._rules is None:
            raise RuntimeError("Call association_rules() before profile_rules().")
        return RuleProfiler(
            self._rules,
            n_clusters=n_clusters,
            max_k=max_k,
            random_state=random_state,
        ).fit()

    def profile_items(
        self,
        n_clusters: Union[int, str] = "auto",
        max_k: int = 8,
        random_state: int = 0,
    ) -> "ItemProfiler":
        """
        Cluster items by their aggregated rule-metric fingerprints.

        Requires ``association_rules()`` to have been called first.

        Parameters
        ----------
        n_clusters : int or "auto"
            Number of clusters. ``"auto"`` selects the best k via silhouette score.
        max_k : int
            Upper bound for k search when ``n_clusters="auto"`` (default 8).
        random_state : int
            Reproducibility seed (default 0).

        Returns
        -------
        ItemProfiler
            A fitted profiler. Inspect ``.clusters`` or call ``.recommend()``.
        """
        from .profiling.item_profiler import ItemProfiler
        if self._rules is None:
            raise RuntimeError("Call association_rules() before profile_items().")
        return ItemProfiler(
            self._rules,
            n_clusters=n_clusters,
            max_k=max_k,
            random_state=random_state,
        ).fit()

    def sql(self, query: str) -> pa.Table: return self.conn.query(query)
    def close(self): self.conn.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __repr__(self) -> str: return f"Duwhal(database={self.conn._database!r})"
