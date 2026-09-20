from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pyarrow as pa


@dataclass
class ModelStats:
    """Lightweight container for model-fit metadata and table statistics."""

    model_name: str
    fit_params: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    table_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_arrow(self) -> pa.Table:
        rows = [
            {
                "model_name": self.model_name,
                "param_name": k,
                "param_value": str(v),
                "duration_ms": self.duration_ms,
                "timestamp": self.timestamp,
                **{f"stat_{sk}": str(sv) for sk, sv in self.table_stats.items()},
            }
            for k, v in self.fit_params.items()
        ]
        if not rows:
            rows = [
                {
                    "model_name": self.model_name,
                    "param_name": None,
                    "param_value": None,
                    "duration_ms": self.duration_ms,
                    "timestamp": self.timestamp,
                    **{f"stat_{sk}": str(sv) for sk, sv in self.table_stats.items()},
                }
            ]
        schema = pa.schema([
            ("model_name", pa.string()),
            ("param_name", pa.string()),
            ("param_value", pa.string()),
            ("duration_ms", pa.float64()),
            ("timestamp", pa.float64()),
        ])
        return pa.Table.from_pylist(rows, schema=schema)


def time_fit(stats: Optional[ModelStats]) -> Any:
    """Decorator that records fit duration in a ModelStats object.

    The wrapped method must accept a *stats* keyword argument.
    """
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            stats_obj = kwargs.get("stats") or getattr(self, "_stats", None)
            start = time.perf_counter()
            try:
                return method(self, *args, **kwargs)
            finally:
                if stats_obj is not None:
                    stats_obj.duration_ms = (time.perf_counter() - start) * 1000
        return wrapper
    return decorator
