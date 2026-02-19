"""
duwhal.datasets.media — Music, playlist, and media content generators.

Datasets for recommendation discovery across genres and content bridging.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional


def generate_playlist_data(seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic playlist data with two genres and a cross-genre bridge.

    Structure:
    - **Rock cluster**: Stairway to Heaven, Comfortably Numb, Whole Lotta Love,
      Wish You Were Here
    - **Jazz cluster**: So What, Blue in Green, Giant Steps, My Favorite Things
    - **Bridge**: A single playlist connects "Wish You Were Here" ↔ "So What",
      allowing multi-hop discovery from Rock to Jazz.

    Columns: ``playlist_id``, ``song_name``

    Returns
    -------
    pd.DataFrame
        14 rows across 7 playlists.

    Example
    -------
    >>> from duwhal.datasets import generate_playlist_data
    >>> df = generate_playlist_data()
    >>> # Rock songs linked within playlists
    >>> rock = df[df["song_name"].str.contains("Stairway|Comfortably")]
    """
    data = [
        ("Playlist_1", "Stairway to Heaven"), ("Playlist_1", "Comfortably Numb"),
        ("Playlist_2", "Stairway to Heaven"), ("Playlist_2", "Whole Lotta Love"),
        ("Playlist_3", "Comfortably Numb"), ("Playlist_3", "Wish You Were Here"),
        ("Playlist_4", "So What"), ("Playlist_4", "Blue in Green"),
        ("Playlist_5", "So What"), ("Playlist_5", "Giant Steps"),
        ("Playlist_6", "Blue in Green"), ("Playlist_6", "My Favorite Things"),
        # Bridge: Classic Rock → Jazz
        ("Playlist_Bridge", "Wish You Were Here"), ("Playlist_Bridge", "So What"),
    ]
    return pd.DataFrame(data, columns=["playlist_id", "song_name"])
