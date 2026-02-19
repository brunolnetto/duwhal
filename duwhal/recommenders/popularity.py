from __future__ import annotations
import pyarrow as pa
from typing import Optional, List
from duwhal.core.connection import DuckDBConnection

class PopularityRecommender:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        strategy: str = "global",
        timestamp_col: Optional[str] = None,
        window_days: int = 30,
    ):
        if strategy not in ["global", "trending"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.conn = conn
        self.table_name = table_name
        self.strategy = strategy
        self.timestamp_col = timestamp_col
        self.window_days = window_days
        self._fitted = False

    def fit(self):
        if self.strategy == "trending" and not self.timestamp_col:
            # check for sort_column
            try:
                self.conn.execute(f"SELECT sort_column FROM {self.table_name} LIMIT 0")
                self.timestamp_col = "sort_column"
            except:
                raise ValueError("timestamp_col required for trending strategy.")

        where_clause = ""
        if self.strategy == "trending":
            # Assume timestamp_col is a date or we can handle it
            where_clause = f"WHERE {self.timestamp_col} >= (SELECT MAX({self.timestamp_col}) FROM {self.table_name}) - INTERVAL {self.window_days} DAY"

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _popularity AS
            SELECT node_id, COUNT(*)::FLOAT / (SELECT COUNT(*) FROM {self.table_name} {where_clause}) AS score
            FROM {self.table_name}
            {where_clause}
            GROUP BY 1
        """)
        self._fitted = True
        return self

    def recommend(self, n: int = 10, exclude_items: Optional[List[str]] = None) -> pa.Table:
        if not self._fitted: self.fit()
        
        exclude_sql = ""
        if exclude_items:
            items_str = ", ".join([f"'{i}'" for i in exclude_items])
            exclude_sql = f"WHERE node_id NOT IN ({items_str})"

        query = f"""
            SELECT 
                node_id AS item_id, 
                score,
                RANK() OVER (ORDER BY score DESC) as rank
            FROM _popularity 
            {exclude_sql}
            ORDER BY score DESC 
            LIMIT {n}
        """
        return self.conn.query(query)
