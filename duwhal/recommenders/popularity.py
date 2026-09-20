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

    def _validate_params(self):
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")
        if self.decay_half_life is not None and self.decay_half_life <= 0:
            raise ValueError("decay_half_life must be > 0")

    def fit(self, stats=None):
        import time
        start = time.perf_counter()
        self._validate_params()
        if self.strategy == "trending" and not self.timestamp_col:
            # check for sort_column
            try:
                self.conn.execute(f"SELECT sort_column FROM {self.table_name} LIMIT 0")
                self.timestamp_col = "sort_column"
            except Exception:
                raise ValueError("timestamp_col required for trending strategy.")

        if self.strategy == "trending":
            where_clause = f"WHERE {self.timestamp_col} >= bounds.max_ts - INTERVAL {self.window_days} DAY"
        else:
            where_clause = ""

        ts_col = self.timestamp_col if self.timestamp_col else "NULL"
        half_life_seconds = (self.decay_half_life or 0) * 86400.0
        if self.decay_half_life and self.timestamp_col:
            score_expr = f"SUM(POWER(0.5, EXTRACT(EPOCH FROM (bounds.max_ts - {self.timestamp_col})) / {half_life_seconds}))"
        else:
            score_expr = "COUNT(DISTINCT set_id)::DOUBLE"

        self.conn.execute(f"""
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
                w.raw_score / (SELECT SUM(raw_score) FROM weighted) AS score
            FROM weighted w
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE _popularity_ranks AS
            SELECT node_id AS item_id, score,
                   ROW_NUMBER() OVER (ORDER BY score DESC, node_id) AS rank
            FROM _popularity
        """)
        self._fitted = True
        duration_ms = (time.perf_counter() - start) * 1000
        if stats is not None:
            stats.duration_ms = duration_ms
            catalog_size = self.conn.execute(f"SELECT COUNT(DISTINCT node_id) FROM {self.table_name}").fetchone()[0]
            num_items = self.conn.execute("SELECT COUNT(*) FROM _popularity").fetchone()[0]
            stats.table_stats = {
                "catalog_size": catalog_size,
                "num_items": num_items,
                "strategy": self.strategy,
                "window_days": self.window_days,
                "decay_half_life": self.decay_half_life,
            }
            self._stats = stats
        return self

    def recommend(self, n: int = 10, exclude_items: Optional[Any] = None) -> pa.Table:
        if not self._fitted: self.fit()

        exclude_items = normalize_seeds(exclude_items) if exclude_items else []
        if exclude_items:
            self.conn.register("_exclude_items", seed_table(exclude_items))
            exclude_sql = "WHERE item_id NOT IN (SELECT node_id FROM _exclude_items)"
        else:
            exclude_sql = ""

        query = f"""
            SELECT item_id, score, rank
            FROM _popularity_ranks
            {exclude_sql}
            ORDER BY rank ASC
            LIMIT {n}
        """
        return self.conn.query(query)
