from .api import Duwhal
from .core.connection import DuckDBConnection
from .core.facets import build_composite_key, build_facet_entities, split_by_facet, merge_recommendation_tables
from .mining.association_rules import AssociationRules
from .mining.frequent_itemsets import FrequentItemsets
from .recommenders.item_cf import ItemCF
from .recommenders.popularity import PopularityRecommender
from .recommenders.graph import GraphRecommender
from .evaluation.metrics import evaluate_recommendations
from .evaluation.splitting import temporal_split, random_split
from .mining.sink_sccs import SinkSCCFinder
from .graph import InteractionGraph
from .temporal import (
    DecayGraphRecommender,
    SlidingWindowGraph,
    TemporalSnapshotDiffer,
    EdgeDelta,
    TemporalSCCTracker,
    SCCSnapshot,
    DirectedTemporalGraph,
)
from .aggregation import (
    NodeTemporalAggregator,
    EdgeLifecycleAggregator,
    GraphSnapshotAggregator,
    CommunityAggregator,
    NodeCorrelationMatrix,
    TemporalAggregationReport,
)
from .datasets import (
    generate_3scc_dataset,
    generate_retail_transactions,
    generate_benchmark_patterns,
    generate_playlist_data,
    generate_genomics_data,
    generate_nlp_corpus,
    generate_filter_bubble_data,
    generate_large_scale_data,
    generate_temporal_interactions,
    generate_directed_sequence_data,
)

def load(data, **kwargs) -> Duwhal:
    engine = Duwhal()
    engine.load(data, **kwargs)
    return engine

def connect(database=":memory:", **kwargs) -> Duwhal:
    return Duwhal(database=database, **kwargs)

__all__ = [
    "Duwhal",
    "load",
    "connect",
    "DuckDBConnection",
    # Facet helpers
    "build_composite_key",
    "build_facet_entities",
    "split_by_facet",
    "merge_recommendation_tables",
    # Mining
    "AssociationRules",
    "FrequentItemsets",
    "ItemCF",
    "PopularityRecommender",
    "GraphRecommender",
    "evaluate_recommendations",
    "temporal_split",
    "random_split",
    "InteractionGraph",
    "SinkSCCFinder",
    # Temporal algorithms
    "DecayGraphRecommender",
    "SlidingWindowGraph",
    "TemporalSnapshotDiffer",
    "EdgeDelta",
    "TemporalSCCTracker",
    "SCCSnapshot",
    "DirectedTemporalGraph",
    # Temporal aggregators
    "NodeTemporalAggregator",
    "EdgeLifecycleAggregator",
    "GraphSnapshotAggregator",
    "CommunityAggregator",
    "NodeCorrelationMatrix",
    "TemporalAggregationReport",
    # Datasets
    "generate_3scc_dataset",
    "generate_retail_transactions",
    "generate_benchmark_patterns",
    "generate_playlist_data",
    "generate_genomics_data",
    "generate_nlp_corpus",
    "generate_filter_bubble_data",
    "generate_large_scale_data",
    "generate_temporal_interactions",
    "generate_directed_sequence_data",
]
