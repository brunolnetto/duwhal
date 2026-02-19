from __future__ import annotations
import duckdb
from pathlib import Path
from typing import Union, Optional, Any
import pyarrow as pa

class DuckDBConnection:
    """Wrapper for DuckDB connection with utility methods."""
    def __init__(
        self,
        database: Union[str, Path] = ":memory:",
        memory_limit: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        self._database = str(database)
        self.conn = duckdb.connect(self._database)
        
        if memory_limit:
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
        if threads:
            self.conn.execute(f"SET threads={threads}")

    def execute(self, query: str, params: Optional[Union[list, dict]] = None) -> duckdb.DuckDBPyConnection:
        return self.conn.execute(query, params)

    def query(self, query: str, params: Optional[Union[list, dict]] = None) -> pa.Table:
        res = self.execute(query, params)
        # Use arrow() which is often mocked in tests
        table = res.arrow()
        if hasattr(table, "read_all"):
             return table.read_all()
        return table

    def table_exists(self, table_name: str) -> bool:
        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 0")
            return True
        except:
            return False

    def register(self, name: str, df: Any):
        self.conn.register(name, df)

    def register_dataframe(self, name: str, df: Any):
        self.register(name, df)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"DuckDBConnection(database={self._database!r})"
