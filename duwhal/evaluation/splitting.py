from __future__ import annotations
from typing import Tuple, Optional, Any
import narwhals as nw
import numpy as np

def _validate_fraction(frac: float):
    if not (0 < frac < 1): raise ValueError("fraction must be in (0, 1)")

def temporal_split(df: Any, timestamp_col: str, test_fraction: float = 0.2, set_col: Optional[str] = None) -> Tuple[Any, Any]:
    _validate_fraction(test_fraction)
    nw_df = nw.from_native(df).sort(timestamp_col)
    
    if set_col:
        # Respect groupings: find the split time that roughly gives test_fraction of sets
        sets = nw_df.select(set_col).unique()
        split_idx = int(len(sets) * (1 - test_fraction))
        pivot_set = sets[split_idx, set_col]
        # Find first occurrence of this set to get a clean split time or just use the set list
        train_sets = sets[:split_idx, set_col]
        return (
            nw_df.filter(nw.col(set_col).is_in(train_sets)).to_native(),
            nw_df.filter(~nw.col(set_col).is_in(train_sets)).to_native()
        )

    split_idx = int(len(nw_df) * (1 - test_fraction))
    return nw_df[:split_idx].to_native(), nw_df[split_idx:].to_native()

def random_split(df: Any, test_fraction: float = 0.2, seed: int = 42, set_col: Optional[str] = None) -> Tuple[Any, Any]:
    _validate_fraction(test_fraction)
    nw_df = nw.from_native(df)
    
    rng = np.random.default_rng(seed)
    if set_col:
        unique_sets = nw_df.select(set_col).unique().to_native()
        # Handle both pandas/polars unique() outputs consistently 
        # (Narwhals .unique() returns a NW DF, .to_native() gives the underlying series/df)
        set_list = nw.from_native(unique_sets).get_column(set_col).to_list()
        rng.shuffle(set_list)
        
        split_idx = int(len(set_list) * (1 - test_fraction))
        train_set = set(set_list[:split_idx])
        
        return (
            nw_df.filter(nw.col(set_col).is_in(train_set)).to_native(),
            nw_df.filter(~nw.col(set_col).is_in(train_set)).to_native()
        )

    indices = np.arange(len(nw_df))
    rng.shuffle(indices)
    split_idx = int(len(nw_df) * (1 - test_fraction))
    return (
        nw_df[indices[:split_idx]].to_native(),
        nw_df[indices[split_idx:]].to_native()
    )
