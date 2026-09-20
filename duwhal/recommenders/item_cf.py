from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import normalize_seeds, seed_table


class ItemCF:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        metric: str = "jaccard",
        min_cooccurrence: int = 1,
        top_k_similar: int = 50,
        shrinkage: int = 0,
    ):
        if metric not in ["jaccard", "cosine", "lift"]:
            raise ValueError(f"Unknown metric: {metric}")
        self.conn = conn
        self.table_name = table_name
        self.metric = metric
        self.min_cooccurrence = min_cooccurrence
        self.top_k_similar = top_k_similar
        self.shrinkage = shrinkage
        self._fitted = False

    def fit(self):
        # Validation for manually set metric
        if self.metric not in ["jaccard", "cosine", "lift"]:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Build co-occurrence matrix and similarity scores
        if self.metric == "jaccard":
            score_expr = "p.cooc / (a_cnt.cnt + b_cnt.cnt - p.cooc)"
        elif self.metric == "cosine":
            score_expr = "p.cooc / (sqrt(a_cnt.cnt) * sqrt(b_cnt.cnt))"
        else:  # lift
            total_n = self.conn.execute(f"SELECT COUNT(DISTINCT set_id) FROM {self.table_name}").fetchone()[0]
            score_expr = f"(p.cooc * {total_n}) / (a_cnt.cnt * b_cnt.cnt)"

        if self.shrinkage:
            score_expr = f"{score_expr} * (p.cooc / (p.cooc + {self.shrinkage}))"

        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _item_counts AS
            SELECT node_id, COUNT(*) AS cnt FROM {self.table_name} GROUP BY 1
        """)

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _item_similarity AS
            WITH directed_pairs AS (
                SELECT
                    LEAST(a.node_id, b.node_id) AS item_a,
                    GREATEST(a.node_id, b.node_id) AS item_b,
                    COUNT(DISTINCT a.set_id) AS cooc
                FROM {self.table_name} a
                JOIN {self.table_name} b
                  ON a.set_id = b.set_id
                 AND a.node_id != b.node_id
                GROUP BY 1, 2
                HAVING cooc >= {self.min_cooccurrence}
            ),
            pairs AS (
                SELECT item_a, item_b, cooc FROM directed_pairs
                UNION ALL
                SELECT item_b AS item_a, item_a AS item_b, cooc FROM directed_pairs
            )
            SELECT
                p.item_a, p.item_b,
                {score_expr} AS score
            FROM pairs p
            JOIN _item_counts a_cnt ON p.item_a = a_cnt.node_id
            JOIN _item_counts b_cnt ON p.item_b = b_cnt.node_id
            QUALIFY row_number() OVER (PARTITION BY p.item_a ORDER BY score DESC) <= {self.top_k_similar}
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE _item_similarity_index AS
            SELECT item_a AS source,
                   list(item_b ORDER BY score DESC) AS neighbors,
                   list(score ORDER BY score DESC) AS scores
            FROM _item_similarity
            GROUP BY item_a
        """)

        self._fitted = True
        return self

    def get_similar_items(self, item_id: str, n: int = 10) -> pa.Table:
        """Return items similar to a single item."""
        if not self._fitted: self.fit()
        self.conn.register("_item_lookup", pa.Table.from_pylist([{"node_id": str(item_id)}]))
        return self.conn.query("""
            SELECT item_b AS item_id, score
            FROM _item_similarity
            WHERE item_a = (SELECT node_id FROM _item_lookup)
            ORDER BY score DESC
            LIMIT $n
        """.replace("$n", str(int(n))))

    def recommend(
        self,
        seed_items: Any,
        n: int = 10,
        exclude_seed: bool = True,
        seed_weights: Optional[dict[str, float]] = None,
    ) -> pa.Table:
        if not self._fitted: self.fit()
        seeds = normalize_seeds(seed_items)
        if not seeds:
            return pa.Table.from_pylist([], schema=pa.schema([("item_id", pa.string()), ("score", pa.float64())]))

        weights = seed_weights if seed_weights else {item: 1.0 for item in seeds}
        table = seed_table({item: weights.get(item, 1.0) for item in seeds}, weight_col="weight")
        self.conn.register("_seeds", table)

        exclude_sql = "AND item_b NOT IN (SELECT node_id FROM _seeds)" if exclude_seed else ""

        # DuckDB supports positional parameters but not for IN-list construction here;
        # table registration is used for seed values to avoid string interpolation.
        query = f"""
            SELECT
                item_b AS item_id,
                SUM(score * s.weight) AS score
            FROM _item_similarity
            JOIN _seeds s ON _item_similarity.item_a = s.node_id
            {exclude_sql}
            GROUP BY 1
            ORDER BY score DESC
            LIMIT {n}
        """
        return self.conn.query(query)
