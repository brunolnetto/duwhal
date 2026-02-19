from __future__ import annotations
import pyarrow as pa
from typing import List, Optional
from duwhal.core.connection import DuckDBConnection

class GraphRecommender:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        min_cooccurrence: int = 1,
        alpha: float = 0.0, # Bayesian Prior
    ):
        self.conn, self.table_name = conn, table_name
        self.min_cooccurrence, self.alpha = min_cooccurrence, alpha
        self._built = False

    def build(self) -> GraphRecommender:
        self.conn.execute(f"CREATE OR REPLACE TEMP TABLE _item_totals AS SELECT node_id, COUNT(*) AS total_interactions FROM {self.table_name} GROUP BY 1")
        self.conn.execute(f"CREATE OR REPLACE TABLE _item_adjacency AS SELECT item_a AS source, list(item_b) AS neighbors, list(cooc) AS weights FROM (SELECT a.node_id AS item_a, b.node_id AS item_b, COUNT(*) AS cooc FROM {self.table_name} a JOIN {self.table_name} b ON a.set_id = b.set_id AND a.node_id != b.node_id GROUP BY 1, 2 HAVING cooc >= {self.min_cooccurrence}) GROUP BY 1")
        self._built = True
        return self

    def get_neighbors(self, item_id: str) -> pa.Table:
        if not self._built: self.build()
        try: self.conn.execute("SELECT 1 FROM _item_edges LIMIT 0")
        except: return self.conn.query(f"SELECT item_b AS neighbor, cooc AS weight FROM (SELECT source, unnest(neighbors) AS item_b, unnest(weights) AS cooc FROM _item_adjacency) WHERE source = '{item_id}'")
        return self.conn.query(f"SELECT target AS neighbor, weight FROM _item_edges WHERE source = '{item_id}'")

    def _prepare_edges(self, scoring: str):
        self.conn.execute("CREATE OR REPLACE TEMP TABLE _item_edges AS SELECT source, unnest(neighbors) AS target, unnest(weights) AS weight FROM _item_adjacency")
        if scoring in ["probability", "path"]:
             self.conn.execute(f"CREATE OR REPLACE TEMP TABLE _item_edges_scored AS SELECT e.*, (e.weight::DOUBLE + {self.alpha}) / (t.total_interactions + {self.alpha} * 100) AS score_val FROM _item_edges e JOIN _item_totals t ON e.source = t.node_id")
        else:
             self.conn.execute("CREATE OR REPLACE TEMP TABLE _item_edges_scored AS SELECT *, weight::DOUBLE AS score_val FROM _item_edges")

    def _build_traversal_query(self, seeds_str, max_depth, min_weight, agg, exclude, n):
        exc_sql = f"AND item NOT IN ({seeds_str})" if exclude else ""
        return f"""
        WITH RECURSIVE traversal(item, strength, depth, path) AS (
            SELECT node_id, 1.0::DOUBLE/count(*) over(), 0, [node_id]
            FROM {self.table_name} WHERE node_id IN ({seeds_str})
            UNION ALL
            SELECT e.target, t.strength * e.score_val, t.depth + 1, list_append(t.path, e.target)
            FROM traversal t
            JOIN _item_edges_scored e ON t.item = e.source
            WHERE t.depth < {max_depth} AND e.weight >= {min_weight} AND NOT list_contains(t.path, e.target)
        )
        SELECT item AS recommended_item, {agg}(strength) AS total_strength, MIN(depth) AS min_hops, arg_max(array_to_string(path, ' -> '), strength) AS reason
        FROM traversal WHERE depth > 0 {exc_sql} GROUP BY 1 ORDER BY total_strength DESC LIMIT {n}
        """

    def recommend(self, seed_items: List[str], max_depth: int = 2, min_weight: int = 1, n: int = 10, exclude_seed: bool = True, scoring: str = "frequency") -> pa.Table:
        if not self._built: self.build()
        self._prepare_edges(scoring)
        if not seed_items: 
            return pa.Table.from_batches([], schema=pa.schema([("recommended_item", pa.string()), ("total_strength", pa.float64()), ("min_hops", pa.int32()), ("reason", pa.string())]))

        q = self._build_traversal_query(", ".join([f"'{s}'" for s in seed_items]), max_depth, min_weight, "MAX" if scoring == "path" else "SUM", exclude_seed, n)
        return self.conn.query(q)

    def score_basket(self, items: List[str]) -> float:
        if not items: return 1.0
        if not self._built: self.build()
        recs = self.recommend(items[:1], max_depth=len(items)+1, exclude_seed=False, scoring="probability", n=1000)
        others = set(items[1:])
        return sum(row["total_strength"] for row in recs.to_pylist() if row["recommended_item"] in others)
