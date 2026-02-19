"""
duwhal.datasets.genomics — Genomics and biomedical co-occurrence generators.

Datasets for gene co-occurrence analysis and mutation trajectory modelling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def generate_genomics_data(
    n_patients: int = 500,
    n_genes: int = 100,
    n_generic_mutations: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic gene expression / mutation co-occurrence data.

    Structure:
    - **BRCA1 ↔ TP53**: 100 patients carry both mutations (strong signal).
    - **Generic mutations**: 1000 randomly distributed gene observations across
      500 patients, creating background noise.

    This mimics a real-world scenario where a few strong co-mutation patterns
    (like BRCA1-TP53 in breast cancer) emerge from noisy high-dimensional data.

    Columns: ``sample_id``, ``gene_id``

    Parameters
    ----------
    n_patients : int
        Number of patients.
    n_genes : int
        Number of distinct generic gene IDs.
    n_generic_mutations : int
        Number of generic mutation observations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        ~1200 rows of patient-gene pairs.

    Example
    -------
    >>> from duwhal.datasets import generate_genomics_data
    >>> df = generate_genomics_data()
    >>> # BRCA1 should strongly co-occur with TP53
    >>> brca1_patients = set(df[df["gene_id"]=="BRCA1"]["sample_id"])
    >>> tp53_patients = set(df[df["gene_id"]=="TP53"]["sample_id"])
    >>> overlap = len(brca1_patients & tp53_patients)
    """
    data = []

    # Strong signal: BRCA1 + TP53 co-mutation
    for i in range(min(100, n_patients)):
        data.append([f"Patient_{i}", "BRCA1"])
        data.append([f"Patient_{i}", "TP53"])

    # Background noise: generic mutations
    rng = np.random.default_rng(seed)
    for i in range(n_generic_mutations):
        patient = f"Patient_{i % n_patients}"
        gene = f"Gene_{rng.integers(0, n_genes)}"
        data.append([patient, gene])

    return pd.DataFrame(data, columns=["sample_id", "gene_id"])
