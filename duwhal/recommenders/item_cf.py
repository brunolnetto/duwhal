from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import (
    normalize_seed_weights,
    normalize_seeds,
    seed_table,
)


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
        self._stats = None
        self._validate_params()

    def _validate_params(self):
        if self.min_cooccurrence < 1:
            raise ValueError("min_cooccurrence must be >= 1")
        if self.top_k_similar < 1:
            raise ValueError("top_k_similar must be >= 1")
        if self.shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")

    def fit(self, stats=None):
        import time
        start = time.perf_counter()
        self._validate_params()
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
            CREATE OR REPLACE TEMP TABLE _distinct_interactions AS
            SELECT DISTINCT set_id, node_id FROM {self.table_name}
        """)

        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _item_counts AS
            SELECT node_id, COUNT(DISTINCT set_id) AS cnt FROM {self.table_name} GROUP BY 1
        """)

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _item_similarity AS
            WITH unordered_pairs AS (
                SELECT
                    a.node_id AS item_a,
                    b.node_id AS item_b,
                    COUNT(DISTINCT a.set_id) AS cooc
                FROM _distinct_interactions a
                JOIN _distinct_interactions b
                  ON a.set_id = b.set_id
                 AND a.node_id < b.node_id
                GROUP BY 1, 2
                HAVING cooc >= {self.min_cooccurrence}
            ),
            pairs AS (
                SELECT item_a, item_b, cooc FROM unordered_pairs
                UNION ALL
                SELECT item_b AS item_a, item_a AS item_b, cooc FROM unordered_pairs
            )
            SELECT
                p.item_a, p.item_b, p.cooc,
                {score_expr} AS score
            FROM pairs p
            JOIN _item_counts a_cnt ON p.item_a = a_cnt.node_id
            JOIN _item_counts b_cnt ON p.item_b = b_cnt.node_id
            QUALIFY row_number() OVER (
                PARTITION BY p.item_a
                ORDER BY score DESC, p.cooc DESC, p.item_b
            ) <= {self.top_k_similar}
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE _item_similarity_index AS
            SELECT item_a AS source,
                   list(item_b ORDER BY score DESC, cooc DESC, item_b) AS neighbors,
                   list(score ORDER BY score DESC, cooc DESC, item_b) AS scores
            FROM _item_similarity
            GROUP BY item_a
        """)

        self._fitted = True
        duration_ms = (time.perf_counter() - start) * 1000
        if stats is not None:
            stats.duration_ms = duration_ms
            catalog_size = self.conn.execute(f"SELECT COUNT(DISTINCT node_id) FROM {self.table_name}").fetchone()[0]
            indexed_items = self.conn.execute("SELECT COUNT(DISTINCT item_a) FROM _item_similarity").fetchone()[0]
            num_pairs = self.conn.execute("SELECT COUNT(*) FROM _item_similarity").fetchone()[0]
            contexts = self.conn.execute(f"SELECT COUNT(DISTINCT set_id) FROM {self.table_name}").fetchone()[0]
            mean_neighbors = self.conn.execute("""
                SELECT COALESCE(AVG(len(neighbors)), 0)
                FROM _item_similarity_index
            """).fetchone()[0]
            max_neighbors = self.conn.execute("""
                SELECT COALESCE(MAX(len(neighbors)), 0)
                FROM _item_similarity_index
            """).fetchone()[0]
            stats.table_stats = {
                "contexts": contexts,
                "catalog_size": catalog_size,
                "indexed_items": indexed_items,
                "similarity_edges": num_pairs,
                "mean_neighbors": mean_neighbors,
                "max_neighbors": max_neighbors,
                "coverage": round(indexed_items / max(catalog_size, 1), 4),
            }
            self._stats = stats
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
        if n < 1:
            raise ValueError("n must be >= 1")

        if seed_weights is not None:
            weights = normalize_seed_weights(seed_weights)
        elif isinstance(seed_items, dict):
            weights = normalize_seed_weights(seed_items)
        else:
            weights = {item: 1.0 for item in seeds}
        weights = {item: weights.get(item, 1.0) for item in seeds}
        table = seed_table(weights, weight_col="weight")
        self.conn.register("_seeds", table)

        # Use the precomputed adjacency-style serving index.
        if exclude_seed:
            exclude_filter = """
                WHERE item_id NOT IN (
                    SELECT node_id FROM _seeds
                )
            """
        else:
            exclude_filter = ""

        query = f"""
            SELECT
                item_id,
                SUM(score * weight) AS score
            FROM (
                SELECT
                    s.weight,
                    unnest(i.neighbors) AS item_id,
                    unnest(i.scores) AS score
                FROM _seeds s
                JOIN _item_similarity_index i ON i.source = s.node_id
            ) sim
            {exclude_filter}
            GROUP BY item_id
            ORDER BY score DESC
            LIMIT {n}
        """
        return self.conn.query(query)

    def recommend_batch(
        self,
        seeds_list: list[Any],
        n: int = 10,
        exclude_seed: bool = True,
        seed_weights_list: Optional[list[dict[str, float]]] = None,
    ) -> pa.Table:
        """True batch inference for multiple seed baskets.

        Returns a single Arrow table with ``basket_id``, ``item_id`` and
        ``score`` columns, ranked per basket.
        """
        if not self._fitted: self.fit()
        if n < 1:
            raise ValueError("n must be >= 1")

        if seed_weights_list is not None and len(seed_weights_list) != len(seeds_list):
            raise ValueError("seed_weights_list must have the same length as seeds_list")

        rows = []
        for basket_id, seeds in enumerate(seeds_list):
            norm_seeds = normalize_seeds(seeds)
            if not norm_seeds:
                continue
            if seed_weights_list is not None and seed_weights_list[basket_id] is not None:
                weights = normalize_seed_weights(seed_weights_list[basket_id])
            elif isinstance(seeds, dict):
                weights = normalize_seed_weights(seeds)
            else:
                weights = {item: 1.0 for item in norm_seeds}
            for item in norm_seeds:
                rows.append({"basket_id": basket_id, "node_id": item, "weight": float(weights.get(item, 1.0))})

        if not rows:
            return pa.Table.from_pylist(
                [],
                schema=pa.schema([
                    ("basket_id", pa.int64()),
                    ("item_id", pa.string()),
                    ("score", pa.float64()),
                ]),
            )

        self.conn.register("_batch_seeds", pa.Table.from_pylist(rows))
        exclude_filter = """
            AND NOT EXISTS (
                SELECT 1
                FROM _batch_seeds seeds
                WHERE seeds.basket_id = rec.basket_id
                  AND seeds.node_id = rec.item_id
            )
        """ if exclude_seed else ""
        query = f"""
            SELECT
                rec.basket_id,
                rec.item_id,
                rec.score
            FROM (
                SELECT
                    bs.basket_id,
                    u.item_id,
                    SUM(u.score * bs.weight) AS score
                FROM _item_similarity_index i
                JOIN _batch_seeds bs ON i.source = bs.node_id
                CROSS JOIN LATERAL (
                    SELECT unnest(i.neighbors) AS item_id, unnest(i.scores) AS score
                ) u
                GROUP BY bs.basket_id, u.item_id
            ) rec
            WHERE 1=1
            {exclude_filter}
            QUALIFY row_number() OVER (PARTITION BY rec.basket_id ORDER BY rec.score DESC, rec.item_id) <= {n}
            ORDER BY rec.basket_id, rec.score DESC, rec.item_id
        """
        return self.conn.query(query)
