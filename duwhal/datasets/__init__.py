"""
duwhal.datasets — Synthetic dataset generators for every domain.

Each generator produces a ready-to-use ``pd.DataFrame`` with documented
columns, known ground-truth patterns, and optional random seeds for
reproducibility.

Domains
-------
- **retail**: Market basket analysis, cross-sell, benchmark patterns
- **media**: Playlist discovery, genre bridging
- **genomics**: Gene co-occurrence, mutation trajectories
- **nlp**: Token co-occurrence, topic clusters, n-grams
- **social**: Filter bubbles, community equilibrium
- **scaling**: Large-scale power-law data for benchmarking
- **scc_synthetic**: Controlled multi-SCC graphs for path-integral research
"""

from .scc_synthetic import generate_3scc_dataset
from .retail import generate_retail_transactions, generate_benchmark_patterns
from .media import generate_playlist_data
from .genomics import generate_genomics_data
from .nlp import generate_nlp_corpus
from .social import generate_filter_bubble_data
from .scaling import generate_large_scale_data

__all__ = [
    # SCC / Path Integral
    "generate_3scc_dataset",
    # Retail
    "generate_retail_transactions",
    "generate_benchmark_patterns",
    # Media
    "generate_playlist_data",
    # Genomics
    "generate_genomics_data",
    # NLP
    "generate_nlp_corpus",
    # Social
    "generate_filter_bubble_data",
    # Scaling
    "generate_large_scale_data",
]
