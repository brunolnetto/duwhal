from __future__ import annotations

from typing import Any, Iterable, List, Optional, Union

import pyarrow as pa


MAX_SQL_INLIST_SIZE = 1000


def normalize_seeds(seeds: Any) -> List[str]:
    """Normalize seed items to a list of unique VARCHAR strings.

    Accepts a single item string, a list of strings, or a dict mapping
    item to weight.  Non-string values are cast to string.  Empty or
    None values are dropped.
    """
    if seeds is None:
        return []
    if isinstance(seeds, dict):
        raw_items = list(seeds.keys())
    elif isinstance(seeds, (str, bytes)):
        raw_items = [seeds]
    else:
        try:
            raw_items = list(seeds)
        except TypeError:
            raw_items = [seeds]
    normalized: List[str] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        normalized.append(str(item).strip())
    # Preserve order while removing duplicates.
    seen = set()
    out: List[str] = []
    for item in normalized:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def seed_weights(seeds: Any) -> dict[str, float]:
    """Return a mapping from normalized seed item to weight.

    Defaults to 1.0 when no weight is provided.
    """
    if isinstance(seeds, dict):
        return {item: float(weight) for item, weight in seeds.items() if weight is not None}
    return {item: 1.0 for item in normalize_seeds(seeds)}


def seed_table(seeds: Any, weight_col: Optional[str] = None) -> pa.Table:
    """Build a PyArrow table suitable for registering as DuckDB temp table _seeds.

    If *weight_col* is provided, the table includes that column populated from
    a dict of seeds; otherwise seeds are treated as unweighted.
    """
    if isinstance(seeds, dict) and weight_col:
        rows = [
            {"node_id": item, weight_col: float(weight)}
            for item, weight in seeds.items()
            if weight is not None
        ]
    else:
        rows = [{"node_id": item} for item in normalize_seeds(seeds)]
    schema = [("node_id", pa.string()), (weight_col, pa.float64())] if weight_col else [("node_id", pa.string())]
    return pa.Table.from_pylist(rows, schema=pa.schema(schema))


def inlist_param(values: List[str]) -> tuple[str, list]:
    """Return a safe positional parameter placeholder string and bound values.

    DuckDB accepts ``?`` positional parameters in IN clauses.  Caller must
    pass the returned *params* alongside the SQL string.
    """
    placeholders = ", ".join("?" for _ in values)
    return f"({placeholders})", list(values)
