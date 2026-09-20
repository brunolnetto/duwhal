from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import normalize_seeds, seed_table


class PopularityRecommender:
    """Popularity recommender measuring context penetration.

    The default score counts ``COUNT(DISTINCT set_id)`` and normalises it to a
    probability distribution. This is **context popularity** (also called
    context penetration): how many distinct contexts contain an item, not how
    many raw events it received.

    When ``decay_half_life`` is set the score is computed as the sum of
    per-event decay weights. Because repeated events inside the same context
    each contribute a decayed weight, the decayed score is **event-based**
    rather than context-based.
    """

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        strategy: str = "global",
        timestamp_col: Optional[str] = None,
        window_days: int = 30,
        decay_half_life: Optional[int] = None,
    ):
        if strategy not in ["global", "trending"]:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.conn = conn
        self.table_name = table_name
        self.strategy = strategy
        self.timestamp_col = timestamp_col
        self.window_days = window_days
        self.decay_half_life = decay_half_life
        self._fitted = False
        self._stats = None

        self._validate_params()

    def _validate_params(self) -> None:
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")

        if self.decay_half_life is not None and self.decay_half_life <= 0:
            raise ValueError("decay_half_life must be > 0")

    @staticmethod
    def _validate_recommend_params(*, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")

    @staticmethod
    def _empty_batch_result() -> pa.Table:
        return pa.Table.from_pylist(
            [],
            schema=pa.schema(
                [
                    ("basket_id", pa.int64()),
                    ("item_id", pa.string()),
                    ("score", pa.float64()),
                    ("rank", pa.int64()),
                ]
            ),
        )

    def _resolve_timestamp_col(self) -> str:
        if self.timestamp_col:
            return self.timestamp_col

        try:
            self.conn.execute(
                f"SELECT sort_column FROM {self.table_name} LIMIT 0"
            )
            return "sort_column"
        except Exception:
            raise ValueError(
                "timestamp_col or sort_column required when "
                "decay_half_life is set."
            )

    def fit(self, stats=None) -> PopularityRecommender:
        import time

        start = time.perf_counter()
        self._validate_params()

        if self.strategy == "trending" and not self.timestamp_col:
            self.timestamp_col = self._resolve_timestamp_col()

        if self.decay_half_life:
            self.timestamp_col = self._resolve_timestamp_col()

        if self.strategy == "trending":
            where_clause = (
                f"WHERE {self.timestamp_col} >= "
                f"bounds.max_ts - INTERVAL {self.window_days} DAY"
            )
        else:
            where_clause = ""

        ts_col = self.timestamp_col if self.timestamp_col else "NULL"
        half_life_seconds = (self.decay_half_life or 0) * 86400.0

        if self.decay_half_life:
            score_expr = (
                "SUM(POWER(0.5, EXTRACT(EPOCH FROM "
                f"(bounds.max_ts - {self.timestamp_col})) "
                f"/ {half_life_seconds}))"
            )
        else:
            score_expr = "COUNT(DISTINCT set_id)::DOUBLE"

        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE _popularity AS

            WITH bounds AS (
                SELECT MAX({ts_col}) AS max_ts
                FROM {self.table_name}
            ),

            weighted AS (
                SELECT
                    node_id,
                    {score_expr} AS raw_score
                FROM {self.table_name}
                CROSS JOIN bounds
                {where_clause}
                GROUP BY 1
            )

            SELECT
                w.node_id,
                w.raw_score
                    / (SELECT SUM(raw_score) FROM weighted)
                    AS score
            FROM weighted w
            """
        )

        self.conn.execute(
            """
            CREATE OR REPLACE TABLE _popularity_ranks AS

            SELECT
                node_id AS item_id,
                score,
                ROW_NUMBER() OVER (
                    ORDER BY score DESC, node_id
                ) AS rank
            FROM _popularity
            """
        )

        self._fitted = True

        duration_ms = (
            time.perf_counter() - start
        ) * 1000

        if stats is not None:
            stats.duration_ms = duration_ms

            catalog_size = self.conn.execute(
                f"SELECT COUNT(DISTINCT node_id) FROM {self.table_name}"
            ).fetchone()[0]

            num_items = self.conn.execute(
                "SELECT COUNT(*) FROM _popularity"
            ).fetchone()[0]

            stats.table_stats = {
                "catalog_size": catalog_size,
                "num_items": num_items,
                "strategy": self.strategy,
                "window_days": self.window_days,
                "decay_half_life": self.decay_half_life,
            }

            self._stats = stats

        return self

    def recommend(
        self,
        n: int = 10,
        exclude_items: Optional[Any] = None,
    ) -> pa.Table:
        if not self._fitted:
            self.fit()

        self._validate_recommend_params(n=n)

        exclusions = (
            normalize_seeds(exclude_items)
            if exclude_items
            else []
        )

        if exclusions:
            self.conn.register(
                "_exclude_items",
                seed_table(exclusions),
            )

            exclude_sql = """
                WHERE item_id NOT IN (
                    SELECT node_id
                    FROM _exclude_items
                )
            """
        else:
            exclude_sql = ""

        query = f"""
            SELECT
                item_id,
                score,
                rank
            FROM _popularity_ranks
            {exclude_sql}
            ORDER BY rank ASC
            LIMIT {int(n)}
        """

        return self.conn.query(query)


    def recommend_batch(
        self,
        exclude_items_list: list[Any],
        n: int = 10,
    ) -> pa.Table:
        """Recommend popular items for multiple baskets in one query.

        ``PopularityRecommender`` does not require seed items.  The batch input
        therefore represents per-basket exclusions.  Each position in
        ``exclude_items_list`` defines one ``basket_id`` and may contain any
        value accepted by :func:`normalize_seeds`, including ``None`` or an
        empty iterable.

        The returned ``rank`` is the original global popularity rank, matching
        scalar :meth:`recommend` semantics after exclusions are applied.
        """
        if not self._fitted:
            self.fit()

        self._validate_recommend_params(n=n)

        if not exclude_items_list:
            return self._empty_batch_result()

        basket_rows = [
            {"basket_id": basket_id}
            for basket_id in range(len(exclude_items_list))
        ]

        exclusion_rows = []

        for basket_id, exclude_items in enumerate(exclude_items_list):
            for item in normalize_seeds(exclude_items):
                exclusion_rows.append(
                    {
                        "basket_id": basket_id,
                        "node_id": item,
                    }
                )

        self.conn.register(
            "_popularity_batch_baskets",
            pa.Table.from_pylist(
                basket_rows,
                schema=pa.schema(
                    [("basket_id", pa.int64())]
                ),
            ),
        )

        self.conn.register(
            "_popularity_batch_exclusions",
            pa.Table.from_pylist(
                exclusion_rows,
                schema=pa.schema(
                    [
                        ("basket_id", pa.int64()),
                        ("node_id", pa.string()),
                    ]
                ),
            ),
        )

        query = f"""
            WITH exclusion_counts AS (
                SELECT
                    b.basket_id,
                    COUNT(e.node_id)::BIGINT AS exclusion_count
                FROM _popularity_batch_baskets b
                LEFT JOIN _popularity_batch_exclusions e
                  ON e.basket_id = b.basket_id
                GROUP BY b.basket_id
            ),

            candidates AS (
                SELECT
                    b.basket_id,
                    p.item_id,
                    p.score,
                    p.rank
                FROM exclusion_counts b
                JOIN _popularity_ranks p
                  ON p.rank <= {int(n)} + b.exclusion_count
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM _popularity_batch_exclusions e
                    WHERE
                        e.basket_id = b.basket_id
                        AND e.node_id = p.item_id
                )
            ),

            ranked AS (
                SELECT
                    basket_id,
                    item_id,
                    score,
                    rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY basket_id
                        ORDER BY rank
                    ) AS basket_rank
                FROM candidates
            )

            SELECT
                basket_id,
                item_id,
                score,
                rank
            FROM ranked
            WHERE basket_rank <= {int(n)}
            ORDER BY
                basket_id,
                rank
        """

        return self.conn.query(query)
