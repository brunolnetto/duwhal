"""
duwhal.datasets.temporal — Synthetic temporal interaction generators.

Produces interaction DataFrames with deliberate time-variant patterns,
suitable for benchmarking all five temporal algorithms in
:mod:`duwhal.temporal` and :mod:`duwhal.aggregation`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def generate_temporal_interactions(
    seed: Optional[int] = None,
    n_weeks: int = 12,
) -> pd.DataFrame:
    """
    Generate synthetic interactions with known temporal structure.

    The timeline is divided into three phases:

    - **Early (weeks 1–4)**: Items A, B, C co-occur in most sessions.
    - **Mid (weeks 5–8)**: Items D, E, F form a new cluster; A–C fade.
    - **Late (weeks 9–12)**: Items G, H, I dominate; D–F sustain weakly.

    Item X acts as a *bridge* node, appearing persistently in all phases.

    Columns: ``session_id``, ``item_id``, ``timestamp``

    Load with duwhal::

        db.load_interactions(df, set_col="session_id",
                             node_col="item_id", sort_col="timestamp")

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    n_weeks : int
        Total timeline width in weeks (must be even; default 12).

    Returns
    -------
    pd.DataFrame
        ~540 rows across ~180 sessions (≈3 items/session on average).

    Examples
    --------
    >>> from duwhal.datasets import generate_temporal_interactions
    >>> df = generate_temporal_interactions(seed=42)
    >>> df.head()
    """
    import random

    rng = random.Random(seed)
    rows: list[tuple[str, str, str]] = []

    base = pd.Timestamp("2024-01-01")
    phase_weeks = n_weeks // 3  # length of each phase in weeks

    phase_items = {
        0: ["A", "B", "C"],  # early cluster
        1: ["D", "E", "F"],  # mid cluster
        2: ["G", "H", "I"],  # late cluster
    }
    bridge = "X"

    session_id = 0
    for week in range(n_weeks):
        phase = week // phase_weeks if week // phase_weeks < 3 else 2
        cluster = phase_items[phase]

        # 15 sessions per week
        for _ in range(15):
            session_id += 1
            sid = f"S{session_id:04d}"

            # Choose 2–3 items from the dominant cluster
            k = rng.randint(2, min(3, len(cluster)))
            items = rng.sample(cluster, k)

            # 40 % of sessions add bridge node X
            if rng.random() < 0.40:
                items.append(bridge)

            # Weak bleed-in from previous phase
            if phase > 0 and rng.random() < 0.15:
                prev_cluster = phase_items[phase - 1]
                items.append(rng.choice(prev_cluster))

            # All items in the session share the same day-level timestamp
            # with per-item offsets so ordering is deterministic
            day_offset = week * 7 + rng.randint(0, 6)
            base_ts = base + pd.Timedelta(days=day_offset)

            for pos, item in enumerate(items):
                ts = base_ts + pd.Timedelta(minutes=pos * 5 + rng.randint(0, 3))
                rows.append((sid, item, ts.strftime("%Y-%m-%d %H:%M:%S")))

    return pd.DataFrame(rows, columns=["session_id", "item_id", "timestamp"])


def generate_directed_sequence_data(
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate interaction data with strong directional sequential patterns.

    Known sequences:
    - A → B → C  (checkout funnel, 80 % transition probability)
    - D → E       (two-step pattern)
    - F → G → H → I  (four-step chain)

    Columns: ``session_id``, ``item_id``, ``timestamp``

    Load with duwhal::

        db.load_interactions(df, set_col="session_id",
                             node_col="item_id", sort_col="timestamp")

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        ~400 rows across ~100 sessions.

    Examples
    --------
    >>> from duwhal.datasets import generate_directed_sequence_data
    >>> df = generate_directed_sequence_data(seed=0)
    >>> len(df)
    400
    """
    import random

    rng = random.Random(seed)
    rows: list[tuple[str, str, str]] = []

    base = pd.Timestamp("2024-03-01")
    sequences = [
        (["A", "B", "C"], 0.80),  # funnel
        (["D", "E"], 0.90),  # two-step
        (["F", "G", "H", "I"], 0.70),  # four-step chain
    ]

    session_id = 0
    for trial in range(100):
        seq, prob = rng.choice(sequences)
        session_id += 1
        sid = f"T{session_id:03d}"

        ts = base + pd.Timedelta(days=trial // 5, hours=rng.randint(0, 23))
        emitted = []
        for pos, item in enumerate(seq):
            if pos == 0 or rng.random() < prob:
                emitted.append(item)
            else:
                break  # truncate the sequence probabilistically
        for pos, item in enumerate(emitted):
            step_ts = ts + pd.Timedelta(minutes=pos * 10)
            rows.append((sid, item, step_ts.strftime("%Y-%m-%d %H:%M:%S")))

    return pd.DataFrame(rows, columns=["session_id", "item_id", "timestamp"])
