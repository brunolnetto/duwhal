"""
duwhal.core.facets
==================
Pure-Python helpers that encode *context facet* enrichment strategies so users
do not have to write the boilerplate themselves.

Two conceptual layers
---------------------
- **Primary context** – the natural interaction container
  (order_id, session_id, playlist_id, …).
- **Facets** – attributes that describe that context
  (region, day_period, campaign, language, store_type, …).

Three transformation helpers
----------------------------
build_composite_key
    Pattern 1 – concatenate primary key + facet columns into a single context
    key column.  Use when behaviour differs so much across slices that
    cross-slice co-occurrence would be noise.

build_facet_entities
    Pattern 3 – inject pseudo-entity rows like ``facet:region=EU`` into each
    context alongside real items.  The graph / rules engine then learns
    associations between items and facet signals without any schema change.

split_by_facet
    Pattern 2 helper – return a dict of {facet_label: subset_df} so callers
    can iterate and run per-slice models without writing the groupby themselves.
    Accepts a single column name or an ordered list of column names.

merge_recommendation_tables
    Blending function for the fallback path in ``recommend_by_facet``: when
    the BFS coarsening finds multiple coarser slices each returning non-empty
    results, this function merges them according to a configurable strategy.

See Also
--------
``Duwhal.recommend_by_facet`` in ``duwhal.api`` for the high-level parallel-run
API built on top of these helpers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_composite_key(
    df: pd.DataFrame,
    set_col: str,
    facet_cols: List[str],
    *,
    sep: str = "|",
    out_col: str = "_context_key",
) -> pd.DataFrame:
    """Return *df* with a new composite context-key column.

    The composite key is ``set_col + sep + facet_col1_value + sep + …``.
    Original columns are left intact so the caller can still inspect them.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "order_id": [1, 1, 2],
    ...     "item":     ["Pasta", "Wine", "Coffee"],
    ...     "region":   ["EU", "EU", "BR"],
    ... })
    >>> build_composite_key(df, "order_id", ["region"], out_col="ctx")
    """
    _validate_columns(df, [set_col] + facet_cols)
    validate_disjoint(set_col=set_col, facet_cols=facet_cols)
    out = df.copy()
    parts = [out[set_col].astype(str)] + [out[c].astype(str) for c in facet_cols]
    out[out_col] = _join_series(parts, sep)
    return out


def build_facet_entities(
    df: pd.DataFrame,
    set_col: str,
    node_col: str,
    facet_cols: List[str],
    *,
    prefix: str = "facet",
    sep: str = "|",
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return an enriched long-form DataFrame ready for ``load_interactions``.

    For every unique (set_col, facet combination) a pseudo-entity row is
    appended, e.g. ``facet:region=EU|facet:day_period=night``.

    Parameters
    ----------
    df:
        Raw interaction DataFrame.
    set_col:
        Primary-context column (e.g. ``"order_id"``).
    node_col:
        Item / entity column (e.g. ``"item"``).
    facet_cols:
        Facet columns to encode as pseudo-entities
        (e.g. ``["region", "day_period"]``).
    prefix:
        Namespace prefix for the pseudo-entity label (default ``"facet"``).
    sep:
        Separator between individual facet tokens in the label (default ``"|"``).
    extra_cols:
        Additional columns to preserve (e.g. ``["timestamp"]``).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns [set_col, node_col] + extra_cols.
    """
    keep = [set_col, node_col] + (extra_cols or [])
    _validate_columns(df, keep + facet_cols)
    validate_disjoint(
        set_col=set_col,
        node_col=node_col,
        facet_cols=facet_cols,
        extra_cols=extra_cols,
    )

    item_rows = df[keep].copy()

    facet_label_parts = [
        df[c].astype(str).apply(lambda v, col=c: f"{prefix}:{col}={v}")
        for c in facet_cols
    ]
    facet_labels = _join_series(facet_label_parts, sep)

    # For facet rows, we preserve extra_cols by picking the first value per set_col
    facet_cols_to_keep = [set_col] + (extra_cols or [])
    facet_rows = (
        df[facet_cols_to_keep]
        .assign(**{node_col: facet_labels})
        .groupby(set_col, as_index=False, sort=False)
        .first()
    )

    return pd.concat([item_rows, facet_rows], ignore_index=True)


def split_by_facet(
    df: pd.DataFrame,
    set_col: str,
    node_col: str,
    facet_cols: Union[str, List[str]],
    *,
    extra_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Split *df* into per-facet sub-DataFrames (pattern 2 helper).

    Returns a mapping ``{facet_label: subset_df}``.  When *facet_cols* is a
    list the label is ``"col1=val1|col2=val2|…"`` for each unique combination.
    Each subset contains only *set_col* and *node_col* (plus any *extra_cols*)
    so it can be passed directly to ``load_interactions``.

    Parameters
    ----------
    facet_cols:
        A single column name or an ordered list of column names to group by.
        When a list is given, each unique value-tuple becomes a slice and the
        key encodes all dimensions: ``"region=EU|day_period=night"``.

    Examples
    --------
    >>> slices = split_by_facet(df, "order_id", "item", ["region", "day_period"])
    >>> for label, sub_df in slices.items():
    ...     with Duwhal() as db:
    ...         db.load_interactions(sub_df, set_col="order_id", node_col="item")
    ...         recs = db.recommend(["Pasta"], strategy="graph")
    """
    if isinstance(facet_cols, str):
        facet_cols = [facet_cols]
    keep_base = [set_col, node_col] + (extra_cols or [])
    _validate_columns(df, keep_base + facet_cols)
    validate_disjoint(
        set_col=set_col,
        node_col=node_col,
        facet_cols=facet_cols,
        extra_cols=extra_cols,
    )
    keep_unique = list(dict.fromkeys(keep_base))

    result: Dict[str, pd.DataFrame] = {}
    for group_vals, group_df in df.groupby(facet_cols, sort=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        label = "|".join(f"{col}={val}" for col, val in zip(facet_cols, group_vals))
        result[label] = group_df[keep_unique].copy()
    return result


def merge_recommendation_tables(
    tables: List[pa.Table],
    strategy: str = "union",
    n: int = 10,
    score_col: Optional[str] = None,
    item_col: str = "recommended_item",
) -> pa.Table:
    """Merge recommendation tables from multiple coarser-level fallback candidates.

    Called by ``recommend_by_facet`` when BFS lattice coarsening finds more
    than one coarser slice each returning non-empty results.

    Parameters
    ----------
    tables:
        Non-empty list of Arrow recommendation tables to blend.
    strategy:
        ``"union"``  – sum scores across tables for matching items, then
        re-rank by descending total (default).  Rewards items that appear
        across multiple coarser slices — a consensus signal.

        ``"best"``   – return the single table whose total score sum is
        highest.  Preserves slice purity; good when slices are semantically
        very different and you want the most confident model to win.

        ``"first"``  – return the first table as-is.  Fully deterministic;
        useful for A/B testing or when slice ordering is a business priority.
    n:
        Maximum rows returned.
    score_col:
        Numeric score column to aggregate.  Auto-detected from the first
        table's schema when ``None``.
    item_col:
        Item-identifier column name (default ``"recommended_item"``).

    Notes
    -----
    The right choice between strategies is domain-specific.  ``"union"`` is a
    safe default because it rewards *consensus*: items appearing in multiple
    coarser slices accumulate higher scores.  Use ``"best"`` when you want the
    highest-confidence slice to dominate, or ``"first"`` when slice ordering
    is an explicit business priority.
    """
    if not tables:
        raise ValueError("merge_recommendation_tables: tables list must be non-empty")
    if len(tables) == 1:
        return tables[0].slice(0, n)

    # Auto-detect the score column from the first table's schema
    if score_col is None:
        _numeric = {pa.float64(), pa.float32(), pa.int64(), pa.int32()}
        for field in tables[0].schema:
            if field.name != item_col and field.type in _numeric:
                score_col = field.name
                break

    if strategy == "first":
        return tables[0].slice(0, n)

    if strategy == "best":
        best = max(
            tables,
            key=lambda t: sum(t.column(score_col).to_pylist()) if score_col else len(t),
        )
        return best.slice(0, n)

    if strategy == "union":
        frames = [t.to_pandas() for t in tables]
        combined = pd.concat(frames, ignore_index=True)
        if score_col and score_col in combined.columns:
            agg = (
                combined.groupby(item_col, as_index=False)[score_col]
                .sum()
                .nlargest(n, score_col)
                .reset_index(drop=True)
            )
            return pa.Table.from_pandas(agg[[item_col, score_col]], preserve_index=False)
        # No numeric column — deduplicate and return
        deduped = combined.drop_duplicates(subset=[item_col]).head(n)
        return pa.Table.from_pandas(deduped[[item_col]], preserve_index=False)

    raise ValueError(
        f"Unknown merge strategy={strategy!r}. Expected 'union', 'best', or 'first'."
    )


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Facet helper: missing columns in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )


def validate_disjoint(**groups: Union[str, List[str], None]) -> None:
    """Ensure that the sets of columns assigned to each role are disjoint."""
    seen: Dict[str, str] = {}
    for role, cols in groups.items():
        if cols is None:
            continue
        cset = {cols} if isinstance(cols, str) else set(cols)
        for c in cset:
            if c in seen:
                raise ValueError(
                    f"Facet helper: column collision. {c!r} is used as both "
                    f"{seen[c]!r} and {role!r}. Columns must be pairwise disjoint."
                )
            seen[c] = role


def _join_series(series_list: List[pd.Series], sep: str) -> pd.Series:
    result = series_list[0]
    for s in series_list[1:]:
        result = result + sep + s
    return result
