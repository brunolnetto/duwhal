"""
duwhal.datasets.social — Social graph and community structure generators.

Datasets for filter bubble detection, ecosystem equilibrium analysis,
and community stability research.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional


def generate_filter_bubble_data(seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic gaming/social graph with two tight communities and
    a transient bridge node.

    Structure:
    - **Retro Gaming sink** (Mario, Zelda, Metroid): 300 interactions
      across 150 sessions, very tightly interconnected.
    - **Modern FPS sink** (Halo, CoD, Battlefield): 300 interactions
      across 150 sessions, very tightly interconnected.
    - **Bridge**: "Generic_Game" co-occurs with "Mario" in only 5 sessions.
      Probability: p(Generic_Game → Mario) = 1.0 (high),
      p(Mario → Generic_Game) ≈ 0.05 (low). This makes Generic_Game
      transient — it "leaks" into the Retro Gaming sink.

    Columns: ``user_id``, ``game_title``

    Returns
    -------
    pd.DataFrame
        ~610 rows across ~155 sessions.

    Example
    -------
    >>> from duwhal.datasets import generate_filter_bubble_data
    >>> df = generate_filter_bubble_data()
    >>> # Mario should appear in many more sessions than Generic_Game
    """
    data = [
        # Community 1: Retro Gaming (Sink SCC)
        ("P1", "Mario"), ("P1", "Zelda"),
        ("P2", "Zelda"), ("P2", "Metroid"),
        ("P3", "Metroid"), ("P3", "Mario"),
    ] * 50

    # Community 2: Modern FPS (Sink SCC)
    data += [
        ("P4", "Halo"), ("P4", "CoD"),
        ("P5", "CoD"), ("P5", "Battlefield"),
        ("P6", "Battlefield"), ("P6", "Halo"),
    ] * 50

    # Transient bridge: Generic_Game → Mario (one-directional)
    for i in range(5):
        data.append((f"Bridge_User_{i}", "Generic_Game"))
        data.append((f"Bridge_User_{i}", "Mario"))

    return pd.DataFrame(data, columns=["user_id", "game_title"])
