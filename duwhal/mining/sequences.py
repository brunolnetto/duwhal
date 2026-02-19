from __future__ import annotations
import pyarrow as pa
from typing import Optional
from duwhal.core.connection import DuckDBConnection

class SequentialPatterns:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        timestamp_col: str = "event_time",
        min_support: float = 0.05,
        max_gap: Optional[int] = None,
    ):
        self.conn = conn
        self.table_name = table_name
        self.timestamp_col = timestamp_col
        self.min_support = min_support
        self.max_gap = max_gap

    def _has_column(self, col: str) -> bool:
        try:
            self.conn.execute(f"SELECT {col} FROM {self.table_name} LIMIT 0")
            return True
        except:
            return False

    def fit(self, **kwargs) -> pa.Table:
        # Fallback for sort_column
        if not self._has_column(self.timestamp_col):
            if self._has_column("sort_column"):
                self.timestamp_col = "sort_column"
            else:
                raise ValueError(f"Column '{self.timestamp_col}' not found.")

        total_n = self.conn.execute(f"SELECT COUNT(DISTINCT set_id) FROM {self.table_name}").fetchone()[0]
        
        # A then B patterns with gap logic
        gap_filter = ""
        if self.max_gap is not None:
             # Gap is number of items between matches. 
             # We need row numbers for this.
             self.conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE _tmp_seq AS
                SELECT *, row_number() OVER (PARTITION BY set_id ORDER BY {self.timestamp_col}) AS pos
                FROM {self.table_name}
             """)
             table = "_tmp_seq"
             gap_filter = f"AND b.pos - a.pos - 1 <= {self.max_gap}"
        else:
             table = self.table_name

        query = f"""
        SELECT 
            a.node_id AS prefix,
            b.node_id AS suffix,
            a.node_id || ' -> ' || b.node_id AS pattern,
            COUNT(DISTINCT a.set_id) AS count,
            COUNT(DISTINCT a.set_id)::DOUBLE / {total_n} AS support
        FROM {table} a
        JOIN {table} b ON a.set_id = b.set_id AND a.{self.timestamp_col} < b.{self.timestamp_col} {gap_filter}
        GROUP BY 1, 2, 3
        HAVING support >= {self.min_support}
        ORDER BY support DESC
        """
        return self.conn.query(query)
