# 🦆🐋 Duwhal

**High-Performance Bipartite Interaction Graph Engine — powered by DuckDB.**

[![codecov](https://codecov.io/gh/brunolnetto/duwhal/graph/badge.svg?token=4N88BXX4Q9)](https://codecov.io/gh/brunolnetto/duwhal/graph/badge.svg?token=4N88BXX4Q9)
[![Python ≥ 3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **Duwhal** treats your data as a bipartite graph — **Contexts** (orders, sessions, sentences, patients) connected to **Entities** (products, genes, tokens, games) — and gives you a complete toolkit to mine patterns, generate recommendations, and detect stable communities, all with the speed of DuckDB.

---

## Table of Contents

- [🦆🐋 Duwhal](#-duwhal)
  - [Table of Contents](#table-of-contents)
  - [Why Duwhal?](#why-duwhal)
  - [Core Concepts](#core-concepts)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [API Overview](#api-overview)
    - [`Duwhal` (main engine)](#duwhal-main-engine)
      - [Loading Data](#loading-data)
      - [Mining](#mining)
      - [Recommendation Strategies](#recommendation-strategies)
      - [Sink SCC Detection](#sink-scc-detection)
    - [`InteractionGraph` (graph interface)](#interactiongraph-graph-interface)
    - [Built-in Datasets](#built-in-datasets)
  - [Use Cases](#use-cases)
  - [Evaluation Toolkit](#evaluation-toolkit)
  - [Architecture](#architecture)
  - [License](#license)

---

## Why Duwhal?

Most recommendation and pattern-mining libraries either:

- ❌ Require you to build a matrix first (memory bottleneck), or
- ❌ Are designed for a single domain (e-commerce only), or
- ❌ Don't give you an explanation for *why* an item was recommended.

**Duwhal does things differently:**

| Feature                   | Duwhal                                                        |
| ------------------------- | ------------------------------------------------------------- |
| Ingestion format          | Parquet, CSV, Pandas, Polars, Arrow — zero-copy via DuckDB    |
| Recommendation strategies | Rules, ItemCF, Graph Path Integral, Popularity                |
| Explainability            | Every recommendation includes the path that generated it      |
| Domain agnosticism        | Retail, Genomics, NLP, Music, Social — all the same API       |
| Community detection       | Tarjan SCC to find "equilibrium" communities & filter bubbles |
| Scale                     | 100k+ transactions in seconds, on-disk DuckDB for larger data |

---

## Core Concepts

Duwhal models interactions as a **bipartite graph**:

```
Context₁ ─── Entity_A
Context₁ ─── Entity_B
Context₂ ─── Entity_A
Context₂ ─── Entity_C
```

From this, it projects a **unipartite co-occurrence graph** where edges carry probabilistic weights derived from interaction frequency. Recommendations are then paths through this graph.

This model is universal:

| Domain   | Context             | Entity          |
| -------- | ------------------- | --------------- |
| Retail   | Order               | Product         |
| Genomics | Patient / Sample    | Gene / Mutation |
| Music    | Playlist            | Song            |
| NLP      | Sentence / Document | Token / Concept |
| Social   | User Session        | Content Item    |

---

## Installation

```bash
pip install duwhal
```

With `uv` (recommended):

```bash
uv add duwhal
```

Optional extras:

```bash
pip install "duwhal[pandas]"   # Pandas support
pip install "duwhal[polars]"   # Polars support
```

---

## Quick Start

```python
from duwhal import Duwhal
from duwhal.datasets import generate_retail_transactions

df = generate_retail_transactions()

with Duwhal() as db:
    # 1. Load your interactions
    db.load_interactions(df, set_col="order_id", node_col="item_name")

    # 2. Mine Association Rules
    rules = db.association_rules(min_support=0.2, min_confidence=0.5)
    print(rules.to_pandas().head())

    # 3. Recommend based on rules
    recs = db.recommend(["Pasta"], strategy="rules", n=3)
    print(recs.column("recommended_item").to_pylist())
    # → ['Tomato Sauce', 'Parmesan', ...]

    # 4. Or use Graph Path Integral for multi-hop discovery
    recs_graph = db.recommend(["iPhone 15"], strategy="graph", n=3)
    print(recs_graph.to_pandas()[["recommended_item", "reason"]])
    # Shows the discovery path for each recommendation
```

---

## API Overview

### `Duwhal` (main engine)

The unified entry point for all operations.

```python
from duwhal import Duwhal

db = Duwhal()                          # in-memory (default)
db = Duwhal(database="store.duckdb")  # persistent
```

#### Loading Data

```python
# From a DataFrame (Pandas or Polars)
db.load_interactions(df, set_col="order_id", node_col="item_id")

# From a Parquet file (zero-copy via DuckDB)
db.load_interactions("transactions.parquet", set_col="order_id", node_col="item_id")

# With a sort column for sequential mining
db.load_interactions(df, set_col="order_id", node_col="item_id", sort_col="timestamp")

# From an interaction matrix (rows = contexts, columns = items)
db.load_interaction_matrix(matrix_df)
```

#### Mining

```python
# Frequent Itemsets
itemsets = db.frequent_itemsets(min_support=0.3)

# Association Rules
rules = db.association_rules(min_support=0.1, min_confidence=0.5, min_lift=1.2)

# Sequential Patterns (requires a timestamp column)
patterns = db.sequential_patterns(timestamp_col="ts", min_support=0.05, max_gap=1)
```

#### Recommendation Strategies

| Strategy    | Method                       | Best For                                            |
| ----------- | ---------------------------- | --------------------------------------------------- |
| `"rules"`   | Association Rules            | High-confidence, interpretable                      |
| `"cf"`      | Item Collaborative Filtering | Similarity-based ("users who liked X also liked Y") |
| `"graph"`   | Path Integral traversal      | Multi-hop discovery, sparse data                    |
| `"popular"` | Global / windowed popularity | Cold-start, trending                                |
| `"auto"`    | Picks the best available     | General use                                         |

```python
# Train models
db.association_rules(min_support=0.1, min_confidence=0.5)
db.fit_cf(metric="jaccard", min_cooccurrence=2)
db.fit_graph(alpha=0.1)
db.fit_popularity(strategy="global")

# Recommend
recs = db.recommend(["item_a"], strategy="cf", n=5)
recs = db.recommend(["item_a"], strategy="graph", scoring="probability", n=5)

# Score a basket's internal cohesion
score = db.score_basket(["Beer", "Diaper"])  # → float
```

#### Sink SCC Detection

Identifies self-sustaining communities — nodes that collectively reinforce each other (Tarjan's algorithm over the probabilistic co-occurrence graph):

```python
sccs = db.find_sink_sccs(min_cooccurrence=5, min_confidence=0.1)
# Returns: node, scc_id, scc_size, is_sink, members
```

---

### `InteractionGraph` (graph interface)

A higher-level, node-centric API for graph analysis tasks.

```python
from duwhal import InteractionGraph

with InteractionGraph() as graph:
    graph.load_interactions(df, context_col="user_id", node_col="game_title")
    graph.build_topology(min_interactions=2)

    # Multi-hop proximity ranking from seed nodes
    results = graph.rank_nodes(["Mario"], steps=3, scoring="probability", limit=5)
    # Returns: node, score, steps, reason (path)

    # Detect Filter Bubbles / Equilibrium Communities
    communities = graph.find_equilibrium_communities(min_cooccurrence=5, min_confidence=0.1)
```

---

### Built-in Datasets

Duwhal ships with synthetic generators for every domain, featuring **known ground-truth patterns** so you can validate algorithms instantly:

```python
from duwhal.datasets import (
    generate_retail_transactions,    # iPhone → Silicone Case, Pasta → Tomato Sauce
    generate_benchmark_patterns,     # Beer & Diaper (100% co-occurrence), Milk/Bread/Butter
    generate_playlist_data,          # Rock cluster ↔ Jazz cluster with bridge
    generate_genomics_data,          # BRCA1 ↔ TP53 co-mutation signal
    generate_nlp_corpus,             # Tech cluster ↔ Economy cluster with bridge sentence
    generate_filter_bubble_data,     # Retro Gaming sink & Modern FPS sink with transient bridge
    generate_large_scale_data,       # Power-law 100k+ transactions for benchmarking
    generate_3scc_dataset,           # Controlled 3-SCC graph for path-integral research
)
```

Each generator returns a `pd.DataFrame` with documented columns and optional seed for reproducibility.

---

## Use Cases

Explore the [`examples/use_cases/`](./examples/use_cases/) directory:

| Example                                                                           | Domain   | Key Technique                                        |
| --------------------------------------------------------------------------------- | -------- | ---------------------------------------------------- |
| [`retail_market_basket.py`](./examples/use_cases/retail_market_basket.py)         | Retail   | Association Rules + Sequential Patterns              |
| [`benchmarking_models.py`](./examples/use_cases/benchmarking_models.py)           | Any      | Model comparison: Rules vs CF vs Graph vs Popularity |
| [`genomics_trajectories.py`](./examples/use_cases/genomics_trajectories.py)       | Genomics | Graph Path Integral over gene co-mutation data       |
| [`nlp_token_cooccurrence.py`](./examples/use_cases/nlp_token_cooccurrence.py)     | NLP      | Token proximity + sequential n-gram discovery        |
| [`media_playlist_discovery.py`](./examples/use_cases/media_playlist_discovery.py) | Music    | Multi-hop cross-genre discovery                      |
| [`ecosystem_equilibrium.py`](./examples/use_cases/ecosystem_equilibrium.py)       | Social   | Sink SCC detection for filter bubble analysis        |
| [`evaluation_scaling.py`](./examples/use_cases/evaluation_scaling.py)             | Any      | Large-scale ingestion + benchmarking on 100k+ rows   |

---

## Evaluation Toolkit

```python
from duwhal.evaluation import temporal_split, random_split, evaluate_recommendations

# Split interactions temporally (respects time ordering)
train, test = temporal_split(df, test_fraction=0.2, timestamp_col="ts")

# Or randomly
train, test = random_split(df, test_fraction=0.2, seed=42)

# Evaluate recommendations
metrics = evaluate_recommendations(model_recs, ground_truth, k=10)
# Returns: precision@k, recall@k, MAP@k
```

---

## Architecture

```
duwhal/
├── api.py                  ← Duwhal: unified engine facade
├── graph.py                ← InteractionGraph: node-centric interface
├── core/
│   ├── connection.py       ← DuckDB connection management
│   └── ingestion.py        ← Multi-format data loading (Parquet, DF, Arrow)
├── mining/
│   ├── frequent_itemsets.py
│   ├── association_rules.py
│   ├── sequences.py        ← Sequential pattern mining
│   └── sink_sccs.py        ← Tarjan SCC + sink identification
├── recommenders/
│   ├── graph.py            ← Path Integral traversal
│   ├── item_cf.py          ← ItemCF (Jaccard / Cosine / Lift)
│   └── popularity.py       ← Global + time-windowed popularity
├── evaluation/
│   ├── metrics.py          ← Precision, Recall, MAP
│   └── splitting.py        ← Temporal and random splits
└── datasets/               ← Synthetic generators for 7 domains
```

---

## License

MIT © Duwhal Contributors
