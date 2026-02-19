"""
duwhal.datasets.nlp — NLP corpus and token co-occurrence generators.

Datasets for semantic proximity, topic modelling, and n-gram discovery.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional


def generate_nlp_corpus(seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic news headline corpus with two topic clusters.

    Structure:
    - **Economics cluster**: Economy, Interest Rates, Inflation, Recession,
      Federal Reserve
    - **Tech cluster**: Tech, AI, Silicon Valley, Nvidia, Generative AI, GPU
    - **Bridge sentence**: One headline mentions both "AI" and "Economy",
      linking the two clusters.

    Positional column ``pos`` enables sequential pattern mining (token ordering).

    Columns: ``sentence_id``, ``token``, ``pos``

    Returns
    -------
    pd.DataFrame
        21 rows across 7 sentences.

    Example
    -------
    >>> from duwhal.datasets import generate_nlp_corpus
    >>> df = generate_nlp_corpus()
    >>> # 'AI' should appear in both Tech and Economy contexts
    >>> ai_sentences = df[df["token"]=="AI"]["sentence_id"].unique()
    """
    headlines = [
        ("S1", "Economy", 1), ("S1", "Interest Rates", 2), ("S1", "Inflation", 3),
        ("S2", "Economy", 1), ("S2", "Interest Rates", 2), ("S2", "Recession", 3),
        ("S3", "Federal Reserve", 1), ("S3", "Interest Rates", 2), ("S3", "Economy", 3),
        ("S4", "Tech", 1), ("S4", "AI", 2), ("S4", "Silicon Valley", 3),
        ("S5", "Tech", 1), ("S5", "AI", 2), ("S5", "Nvidia", 3),
        ("S6", "Generative AI", 1), ("S6", "AI", 2), ("S6", "GPU", 3),
        # Bridge: AI × Economy
        ("S7", "AI", 1), ("S7", "Economy", 2), ("S7", "Jobs", 3),
    ]
    return pd.DataFrame(headlines, columns=["sentence_id", "token", "pos"])
