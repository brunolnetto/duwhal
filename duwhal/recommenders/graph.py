from __future__ import annotations

from typing import Any, List, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import normalize_seeds, seed_table


class GraphRecommender:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        min_cooccurrence: int = 1,
        alpha: float = 0.0,  # Bayesian Prior
    ):
        self.conn, self.table_name = conn, table_name
        self.min_cooccurrence, self.alpha = min_cooccurrence, alpha
        self._built = False
        self._prepared_scoring: str | None = None
        self._prepare_edges_calls = 0
        self._stats = None

    @property
    def prepare_edges_calls(self) -> int:
        return self._prepare_edges_calls

    def _validate_params(self):
        if self.min_cooccurrence < 1:
            raise ValueError("min_cooccurrence must be >= 1")

    def build(self, stats=None) -> GraphRecommender:
        import time
        self._validate_params()
        start = time.perf_counter()
        self.conn.execute(f"CREATE OR REPLACE TEMP TABLE _item_totals AS SELECT node_id, COUNT(DISTINCT set_id) AS total_interactions FROM {self.table_name} GROUP BY 1")

        # Unordered co-occurrence computed once, then expanded to both directions.
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _item_unordered_pairs AS
            SELECT
                a.node_id AS item_a,
                b.node_id AS item_b,
                COUNT(DISTINCT a.set_id) AS cooc
            FROM {self.table_name} a
            JOIN {self.table_name} b
              ON a.set_id = b.set_id
             AND a.node_id < b.node_id
            GROUP BY 1, 2
            HAVING cooc >= {self.min_cooccurrence}
        """)

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _item_adjacency AS
            SELECT
                source,
                list(target ORDER BY cooc DESC) AS neighbors,
                list(cooc ORDER BY cooc DESC) AS weights
            FROM (
                SELECT item_a AS source, item_b AS target, cooc FROM _item_unordered_pairs
                UNION ALL
                SELECT item_b AS source, item_a AS target, cooc FROM _item_unordered_pairs
            )
            GROUP BY 1
        """)

        prior_size = self._prior_catalog_size()
        # Precompute edge scores for both frequency and probability/path modes.
        self.conn.execute("""
            CREATE OR REPLACE TABLE _item_edges_scored AS
            SELECT
                source,
                unnest(neighbors) AS target,
                unnest(weights) AS weight,
                weight::DOUBLE AS frequency_score,
                (weight::DOUBLE + ?) / (t.total_interactions + ? * ?) AS probability_score
            FROM _item_adjacency
            JOIN _item_totals t ON source = t.node_id
        """, [self.alpha, self.alpha, prior_size])

        self._built = True
        self._prepared_scoring = "frequency"
        duration_ms = (time.perf_counter() - start) * 1000
        if stats is not None:
            stats.duration_ms = duration_ms
            stats.table_stats = {
                "num_nodes": self.conn.execute("SELECT COUNT(*) FROM _item_adjacency").fetchone()[0],
                "num_edges": self.conn.execute("SELECT SUM(len(neighbors)) FROM _item_adjacency").fetchone()[0],
            }
            self._stats = stats
        return self

    def _prior_catalog_size(self) -> int:
        """Number of distinct items used as the Bayesian prior denominator."""
        total = self.conn.execute(f"SELECT COUNT(DISTINCT node_id) FROM {self.table_name}").fetchone()[0]
        return max(total, 1)

    def get_neighbors(self, item_id: str) -> pa.Table:
        if not self._built: self.build()
        self.conn.register("_item_lookup", pa.Table.from_pylist([{"node_id": str(item_id)}]))
        return self.conn.query("""
            SELECT target AS neighbor, weight
            FROM _item_edges_scored
            WHERE source = (SELECT node_id FROM _item_lookup)
        """)

    @staticmethod
    def _score_column(scoring: str) -> str:
        if scoring in ["probability", "path"]:
            return "probability_score"
        return "frequency_score"

    def _build_traversal_query(
        self,
        max_depth: int,
        min_weight: int,
        agg: str,
        exclude: bool,
        n: int,
        return_paths: bool,
        beam_width: Optional[int],
        max_expansions: Optional[int],
    ) -> str:
        score_col = self._score_column(agg if agg == "path" else "frequency")
        if return_paths:
            reason_col = ", arg_max(array_to_string(path, ' -> '), strength) AS reason"
            frontier_cols = "item, strength, depth, path"
            seed_select = "SELECT s.node_id AS item, 1.0::DOUBLE / COUNT(*) OVER (), 0, [s.node_id]"
            expand_select = f"SELECT e.target, t.strength * e.{score_col}, t.depth + 1, list_append(t.path, e.target)"
            cycle_filter = "AND NOT list_contains(t.path, e.target)"
        else:
            reason_col = ""
            frontier_cols = "item, strength, depth"
            seed_select = "SELECT s.node_id AS item, 1.0::DOUBLE / COUNT(*) OVER (), 0"
            expand_select = f"SELECT e.target, t.strength * e.{score_col}, t.depth + 1"
            cycle_filter = ""

        exc_sql = "AND item NOT IN (SELECT node_id FROM _seeds)" if exclude else ""
        beam_sql = ""
        if beam_width:
            beam_sql = f"QUALIFY row_number() OVER (PARTITION BY depth ORDER BY strength DESC) <= {beam_width}"
        expansion_sql = ""
        if max_expansions:
            expansion_sql = f"AND (SELECT COUNT(*) FROM traversal) < {max_expansions}"

        return f"""
        WITH RECURSIVE traversal({frontier_cols}) AS (
            {seed_select}
            FROM _seeds s
            WHERE EXISTS (SELECT 1 FROM _item_totals i WHERE i.node_id = s.node_id)
            UNION ALL
            {expand_select}
            FROM traversal t
            JOIN _item_edges_scored e ON t.item = e.source
            WHERE t.depth < {max_depth}
              AND e.weight >= {min_weight}
              {cycle_filter}
              {expansion_sql}
            {beam_sql}
        )
        SELECT item AS recommended_item, {agg}(strength) AS total_strength, MIN(depth) AS min_hops {reason_col}
        FROM traversal
        WHERE depth > 0 {exc_sql}
        GROUP BY item
        ORDER BY total_strength DESC
        LIMIT {n}
        """

    def recommend(
        self,
        seed_items: Any,
        max_depth: int = 2,
        min_weight: int = 1,
        n: int = 10,
        exclude_seed: bool = True,
        scoring: str = "frequency",
        return_paths: bool = False,
        beam_width: Optional[int] = None,
        max_expansions: Optional[int] = None,
    ) -> pa.Table:
        if not self._built: self.build()
        self._validate_params()
        if n < 1:
            raise ValueError("n must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_weight < 0:
            raise ValueError("min_weight must be >= 0")
        seeds = normalize_seeds(seed_items)
        if not seeds:
            schema = [
                ("recommended_item", pa.string()),
                ("total_strength", pa.float64()),
                ("min_hops", pa.int32()),
            ]
            if return_paths:
                schema.append(("reason", pa.string()))
            return pa.Table.from_batches([], schema=pa.schema(schema))

        self.conn.register("_seeds", seed_table(seeds))
        q = self._build_traversal_query(
            max_depth=max_depth,
            min_weight=min_weight,
            agg="MAX" if scoring == "path" else "SUM",
            exclude=exclude_seed,
            n=n,
            return_paths=return_paths,
            beam_width=beam_width,
            max_expansions=max_expansions,
        )
        return self.conn.query(q)

    def score_basket(self, items: List[str]) -> float:
        if not items: return 1.0
        if not self._built: self.build()
        recs = self.recommend(items[:1], max_depth=len(items)+1, exclude_seed=False, scoring="probability", n=1000)
        others = set(items[1:])
        return sum(row["total_strength"] for row in recs.to_pylist() if row["recommended_item"] in others)
