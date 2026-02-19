from __future__ import annotations
from typing import Union, Optional, Any, Callable, List
from pathlib import Path
import pyarrow as pa
import narwhals as nw
from duwhal.core.connection import DuckDBConnection

def _check_column_exists(conn: DuckDBConnection, table_name: str, column_name: str) -> bool:
    try:
        conn.execute(f"SELECT {column_name} FROM {table_name} LIMIT 0")
        return True
    except: return False

def _create_empty_tmp_table(conn: DuckDBConnection):
    conn.execute("DROP VIEW IF EXISTS _tmp_interactions; DROP TABLE IF EXISTS _tmp_interactions;")
    conn.execute("CREATE TEMP TABLE _tmp_interactions (set_id VARCHAR, node_id VARCHAR)")

def _get_rename_map(df, set_col, node_col, sort_col):
    m = {set_col: "set_id", node_col: "node_id"}
    if sort_col: m[sort_col] = "sort_column"
    try: schema = df.collect_schema()
    except: schema = df.columns
    return {k: v for k, v in m.items() if k in schema and k != v}

def _apply_sort_callback(df, cb):
    if not cb: return df
    try: return nw.from_native(cb(df.to_native()))
    except: return df

def _rename_and_sort_df(df, set_col, node_col, sort_col, sort_callback):
    df = df.rename(_get_rename_map(df, set_col, node_col, sort_col))
    df = _apply_sort_callback(df, sort_callback)
    return df.drop_nulls(subset=["set_id", "node_id"])

def _register_df(conn, df):
    try: conn.register_dataframe("_tmp_interactions", df.to_native())
    except Exception as e:
        if "Need a DataFrame" in str(e): _create_empty_tmp_table(conn)
        else: raise

def _prepare_df_source(conn, source, set_col, node_col, sort_col, sort_callback):
    try: df = nw.from_native(source)
    except: return _create_empty_tmp_table(conn)
    try: schema = df.collect_schema()
    except: schema = df.columns
    if not schema: return _create_empty_tmp_table(conn)
    if set_col not in schema or node_col not in schema: raise ValueError("Missing columns")
    _register_df(conn, _rename_and_sort_df(df, set_col, node_col, sort_col, sort_callback))

def _prepare_file_source(conn, source, set_col, node_col, sort_col):
    p = str(source)
    if p.endswith(".csv"): conn.execute(f"CREATE OR REPLACE TEMP TABLE _tmp_raw AS SELECT * FROM read_csv_auto('{p}')")
    elif p.endswith(".parquet"): conn.execute(f"CREATE OR REPLACE TEMP TABLE _tmp_raw AS SELECT * FROM read_parquet('{p}')")
    else: raise ValueError("Unsupported file type")
    e = f", {sort_col} AS sort_column" if sort_col else ""
    conn.execute(f"CREATE OR REPLACE TEMP TABLE _tmp_interactions AS SELECT {set_col}::VARCHAR AS set_id, {node_col}::VARCHAR AS node_id {e} FROM _tmp_raw WHERE {set_col} IS NOT NULL AND {node_col} IS NOT NULL")

def _create_new_table(conn, table_name, has_sort):
    e = ", sort_column" if has_sort else ""
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT set_id::VARCHAR AS set_id, node_id::VARCHAR AS node_id {e} FROM _tmp_interactions")

def _append_to_table(conn, table_name, has_sort):
    if has_sort and not _check_column_exists(conn, table_name, "sort_column"):
        res = conn.execute("SELECT typeof(sort_column) FROM _tmp_interactions LIMIT 1").fetchone()
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN sort_column {res[0] if res else 'VARCHAR'}")
    conn.execute(f"INSERT INTO {table_name} BY NAME SELECT * FROM _tmp_interactions")

def _handle_table_upsert(conn, table_name, append, has_sort):
    if not append or not conn.table_exists(table_name):
        _create_new_table(conn, table_name, has_sort)
    else:
        _append_to_table(conn, table_name, has_sort)

def load_interactions(conn, source, set_col="set_id", node_col="node_id", sort_col=None, sort_callback=None, table_name="interactions", append=False):
    if isinstance(source, (str, Path)): _prepare_file_source(conn, source, set_col, node_col, sort_col)
    else: _prepare_df_source(conn, source, set_col, node_col, sort_col, sort_callback)
    _handle_table_upsert(conn, table_name, append, _check_column_exists(conn, "_tmp_interactions", "sort_column"))
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_set ON {table_name}(set_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_node ON {table_name}(node_id)")
    return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

def _native_filt(df):
    if hasattr(df, "filter") and hasattr(df, "collect_schema"): return df.filter(df["val"] > 0).drop("val")
    return df[df["val"] > 0].drop(columns=["val"])

def _unpivot_fallback(nw_df, set_col):
    native = nw_df.to_native()
    if hasattr(native, "melt"): long = native.melt(id_vars=[set_col], var_name="node_id", value_name="val")
    elif hasattr(native, "unpivot"): long = native.unpivot(index=[set_col], variable_name="node_id", value_name="val")
    else: raise ValueError("Unsupported DataFrame type")
    try: return nw.from_native(long).filter(nw.col("val") > 0).drop("val").to_native()
    except: return _native_filt(long)

def _reset_index(df, sc):
    try:
        ndf = df.copy()
        if not getattr(ndf.index, "name", None): ndf.index.name = sc
        return nw.from_native(ndf.reset_index())
    except: raise ValueError("Input DataFrame must have a 'set_id' column or index")

def _check_nw_df(nw_df):
    if nw_df is not None:
        try: schema = nw_df.collect_schema()
        except: schema = nw_df.columns
        if "set_id" in schema: return True
    return False

def _resolve_set_col(df, nw_df):
    if _check_nw_df(nw_df): return "set_id", nw_df
    if isinstance(df, (str, bytes)) or not hasattr(df, "index"):
        raise ValueError("Input DataFrame must have a 'set_id' column or index")
    sc = getattr(df.index, "name", None) or "set_id"
    return sc, _reset_index(df, sc)

def load_interaction_matrix(conn, df, table_name="interactions", append=False):
    try: nw_df = nw.from_native(df)
    except: nw_df = None
    set_col, nw_df = _resolve_set_col(df, nw_df)
    if nw_df is None: return 0
    try: long = nw_df.unpivot(index=[set_col], variable_name="node_id", value_name="val").filter(nw.col("val") > 0).drop("val").to_native()
    except: long = _unpivot_fallback(nw_df, set_col)
    return load_interactions(conn, long, set_col=set_col, node_col="node_id", table_name=table_name, append=append)
