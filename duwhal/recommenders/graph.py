from __future__ import annotations

from typing import Any, List

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

    def build(self, stats=None) -> GraphRecommender:
        import time
        start = time.perf_counter()
        self.conn.execute(f"CREATE OR REPLACE TEMP TABLE _item_totals AS SELECT node_id, COUNT(DISTINCT set_id) AS total_interactions FROM {self.table_name} GROUP BY 1")
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _item_adjacency AS
            SELECT
                source,
                list(target ORDER BY cooc DESC) AS neighbors,
                list(cooc ORDER BY cooc DESC) AS weights
            FROM (
                SELECT
                    LEAST(a.node_id, b.node_id) AS source,
                    GREATEST(a.node_id, b.node_id) AS target,
                    COUNT(DISTINCT a.set_id) AS cooc
                FROM {self.table_name} a
                JOIN {self.table_name} b
                  ON a.set_id = b.set_id
                 AND a.node_id != b.node_id
                GROUP BY 1, 2
                HAVING cooc >= {self.min_cooccurrence}
                UNION ALL
                SELECT
                    GREATEST(a.node_id, b.node_id) AS source,
                    LEAST(a.node_id, b.node_id) AS target,
                    COUNT(DISTINCT a.set_id) AS cooc
                FROM {self.table_name} a
                JOIN {self.table_name} b
                  ON a.set_id = b.set_id
                 AND a.node_id != b.node_id
                GROUP BY 1, 2
                HAVING cooc >= {self.min_cooccurrence}
            )
            GROUP BY 1
        """)
        self._built = True
        self._prepared_scoring = None
        duration_ms = (time.perf_counter() - start) * 1000
        if stats is not None:
            stats.duration_ms = duration_ms
            stats.table_stats = {
                "num_nodes": self.conn.execute("SELECT COUNT(*) FROM _item_adjacency").fetchone()[0],
                "num_edges": self.conn.execute("SELECT SUM(len(neighbors)) FROM _item_adjacency").fetchone()[0],
            }
            self._stats = stats
        return self

    def get_neighbors(self, item_id: str) -> pa.Table:
        if not self._built: self.build()
        self.conn.register("_item_lookup", pa.Table.from_pylist([{"node_id": str(item_id)}]))
        try:
            self.conn.execute("SELECT 1 FROM _item_edges_scored LIMIT 0")
        except Exception:
            return self.conn.query("""
                SELECT item_b AS neighbor, cooc AS weight
                FROM (
                    SELECT source, unnest(neighbors) AS item_b, unnest(weights) AS cooc
                    FROM _item_adjacency
                )
                WHERE source = (SELECT node_id FROM _item_lookup)
            """)
        return self.conn.query("""
            SELECT target AS neighbor, weight
            FROM _item_edges_scored
            WHERE source = (SELECT node_id FROM _item_lookup)
        """)

    def _prepare_edges(self, scoring: str):
        self._prepare_edges_calls += 1
        self.conn.execute("CREATE OR REPLACE TEMP TABLE _item_edges AS SELECT source, unnest(neighbors) AS target, unnest(weights) AS weight FROM _item_adjacency")
        if scoring in ["probability", "path"]:
             self.conn.execute(f"CREATE OR REPLACE TEMP TABLE _item_edges_scored AS SELECT e.*, (e.weight::DOUBLE + {self.alpha}) / (t.total_interactions + {self.alpha} * 100) AS score_val FROM _item_edges e JOIN _item_totals t ON e.source = t.node_id")
        else:
             self.conn.execute("CREATE OR REPLACE TEMP TABLE _item_edges_scored AS SELECT *, weight::DOUBLE AS score_val FROM _item_edges")
        self._prepared_scoring = scoring

    def _ensure_edges_prepared(self, scoring: str) -> None:
        if self._prepared_scoring == scoring:
            return
        self._prepare_edges(scoring)

    def _build_traversal_query(self, max_depth: int, min_weight: int, agg: str, exclude: bool, n: int, return_paths: bool) -> str:
        reason_col = ", arg_max(array_to_string(path, ' -> '), strength) AS reason" if return_paths else ""
        exc_sql = "AND item NOT IN (SELECT node_id FROM _seeds)" if exclude else ""
        return f"""
        WITH RECURSIVE traversal(item, strength, depth, path) AS (
            SELECT s.node_id, 1.0::DOUBLE / COUNT(*) OVER (), 0, [s.node_id]
            FROM _seeds s
            JOIN {self.table_name} t ON t.node_id = s.node_id
            UNION ALL
            SELECT e.target, t.strength * e.score_val, t.depth + 1, list_append(t.path, e.target)
            FROM traversal t
            JOIN _item_edges_scored e ON t.item = e.source
            WHERE t.depth < {max_depth}
              AND e.weight >= {min_weight}
              AND NOT list_contains(t.path, e.target)
        )
        SELECT item AS recommended_item, {agg}(strength) AS total_strength, MIN(depth) AS min_hops {reason_col}
        FROM traversal
        WHERE depth > 0 {exc_sql}
        GROUP BY 1
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
    ) -> pa.Table:
        if not self._built: self.build()
        self._ensure_edges_prepared(scoring)
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
        )
        return self.conn.query(q)

    def score_basket(self, items: List[str]) -> float:
        if not items: return 1.0
        if not self._built: self.build()
        recs = self.recommend(items[:1], max_depth=len(items)+1, exclude_seed=False, scoring="probability", n=1000)
        others = set(items[1:])
        return sum(row["total_strength"] for row in recs.to_pylist() if row["recommended_item"] in others)
