from __future__ import annotations
from typing import List, Optional, Any
import pyarrow as pa
import narwhals as nw
from duwhal.api import Duwhal

class InteractionGraph:
    """Universal Graph Interface."""
    def __init__(self, database: str = ":memory:"):
        self.db = Duwhal(database=database)

    def __enter__(self):
        self.db.__enter__()
        return self

    def __exit__(self, *_):
        self.db.__exit__(*_)

    def load_interactions(self, data: Any, context_col: str, node_col: str) -> int:
        return self.db.load(data, set_col=context_col, node_col=node_col)

    def build_topology(self, min_interactions: int = 1):
        self.db.fit_graph(min_cooccurrence=min_interactions)
        return self

    def rank_nodes(self, seed_nodes: List[str], steps: int = 2, scoring: str = "probability", limit: int = 10) -> Any:
        table = self.db.recommend(seed_items=seed_nodes, strategy="graph", max_depth=steps, scoring=scoring, n=limit)
        nw_df = nw.from_native(table)
        nw_df = nw_df.rename({"recommended_item": "node", "total_strength": "score", "min_hops": "steps"})
        return nw_df.to_native()

    def find_equilibrium_communities(self, min_cooccurrence: int = 5, min_confidence: float = 0.0) -> Any:
        """Identifies Sink Strongly Connected Components (Equilibrium Sets)."""
        return self.db.find_sink_sccs(min_cooccurrence=min_cooccurrence, min_confidence=min_confidence)
