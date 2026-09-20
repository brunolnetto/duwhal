from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import normalize_seeds, seed_table


class PopularityRecommender:
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

    def fit(self, stats=None):
        import time
        start = time.perf_counter()
        if self.strategy == "trending" and not self.timestamp_col:
            # check for sort_column
            try:
                self.conn.execute(f"SELECT sort_column FROM {self.table_name} LIMIT 0")
                self.timestamp_col = "sort_column"
            except Exception:
                raise ValueError("timestamp_col required for trending strategy.")

        where_clause = ""
        if self.strategy == "trending":
            # Assume timestamp_col is a date or we can handle it
            where_clause = f"WHERE {self.timestamp_col} >= (SELECT MAX({self.timestamp_col}) FROM {self.table_name}) - INTERVAL {self.window_days} DAY"

        score_expr = "COUNT(*)::DOUBLE"
        if self.decay_half_life and self.timestamp_col:
            score_expr = f"SUM(POWER(0.5, EXTRACT(EPOCH FROM ((SELECT MAX({self.timestamp_col}) FROM {self.table_name}) - {self.timestamp_col})) / ({self.decay_half_life} * 86400.0)))"

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _popularity AS
            SELECT
                node_id,
                {score_expr} / (SELECT SUM(cnt) FROM (
                    SELECT node_id, {score_expr} AS cnt
                    FROM {self.table_name}
                    {where_clause}
                    GROUP BY 1
                )) AS score
            FROM {self.table_name}
            {where_clause}
            GROUP BY 1
        """)

        self.conn.execute("""
            CREATE OR REPLACE TABLE _popularity_ranks AS
            SELECT node_id AS item_id, score,
                   RANK() OVER (ORDER BY score DESC) AS rank
            FROM _popularity
        """)
        self._fitted = True
        duration_ms = (time.perf_counter() - start) * 1000
        if stats is not None:
            stats.duration_ms = duration_ms
            stats.table_stats = {
                "num_items": self.conn.execute("SELECT COUNT(*) FROM _popularity").fetchone()[0],
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
