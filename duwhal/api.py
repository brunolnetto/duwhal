from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
import pyarrow as pa
from duwhal.core.connection import DuckDBConnection
from duwhal.core.ingestion import load_interactions, load_interaction_matrix
from duwhal.mining.association_rules import AssociationRules
from duwhal.mining.frequent_itemsets import FrequentItemsets
from duwhal.mining.sequences import SequentialPatterns
from duwhal.recommenders.item_cf import ItemCF
from duwhal.recommenders.popularity import PopularityRecommender
from duwhal.recommenders.graph import GraphRecommender

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

    def load_interactions(self, *args, **kwargs) -> int:
        return load_interactions(self.conn, *args, table_name=self.table_name, **kwargs)

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

    def sql(self, query: str) -> pa.Table: return self.conn.query(query)
    def close(self): self.conn.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __repr__(self) -> str: return f"Duwhal(database={self.conn._database!r})"
