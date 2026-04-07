"""
Cross-Domain Use Case: Context Enrichment via Facets.

Demonstrates three patterns for adding domain-specific signals (region,
day_period, channel, device, season, …) while preserving the universal
duwhal API — **without any user-defined boilerplate**.

Two conceptual layers
---------------------
- **Primary context**: the natural interaction container
  (order_id, session_id, playlist_id, patient_id, …).
- **Facets**: attributes that *describe* that context
  (region, day_period, campaign, language, store_type, …).

Three patterns covered
-----------------------
1. **Composite context key** – ``facet_mode="composite"`` on ``load_interactions``.
2. **Parallel runs by facet** – ``db.recommend_by_facet()``.
3. **Facet-as-entity prefix** – ``facet_mode="entity"`` on ``load_interactions``.

Rule of thumb
-------------
Start with the global model → add facets incrementally → monitor coverage /
recall drift.  Too many facets over-segment data and weaken co-occurrence
signals.
"""

import pandas as pd

from duwhal import Duwhal

# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

RAW = pd.DataFrame(
    {
        "order_id":  [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "item":      [
            "Pasta", "Tomato Sauce", "Olive Oil",
            "Coffee", "Croissant",
            "Pasta", "Olive Oil", "Wine",
            "Coffee", "Granola Bar",
        ],
        "region":     ["EU", "EU", "EU", "BR", "BR", "EU", "EU", "EU", "BR", "BR"],
        "day_period": ["night", "night", "night", "morning", "morning",
                       "night", "night", "night", "morning", "morning"],
    }
)

FACETS = ["region", "day_period"]


# ---------------------------------------------------------------------------
# Pattern 1 — Composite context key  (facet_mode="composite")
# ---------------------------------------------------------------------------

def run_composite_key():
    print("\n" + "=" * 60)
    print("Pattern 1 — Composite context key  (strict segmentation)")
    print("=" * 60)

    with Duwhal() as db:
        db.load_interactions(
            RAW,
            set_col="order_id",
            node_col="item",
            facet_cols=FACETS,
            facet_mode="composite",   # ← the only new argument
        )
        recs = db.recommend(["Pasta"], strategy="graph", n=5)

    print("Seed: ['Pasta']")
    print("Recommendations (graph, composite key):")
    for row in recs.to_pylist():
        print(f"  - {row['recommended_item']:20}  score={row['total_strength']:.4f}")


# ---------------------------------------------------------------------------
# Pattern 2 — Parallel runs by facet  (recommend_by_facet)
# ---------------------------------------------------------------------------

def run_parallel_facets():
    print("\n" + "=" * 60)
    print("Pattern 2 — Parallel runs by facet  (global + per-slice)")
    print("=" * 60)

    with Duwhal() as db:
        results = db.recommend_by_facet(
            ["Coffee"],
            RAW,
            set_col="order_id",
            node_col="item",
            facet_cols="region",   # ← single string or list both accepted
            strategy="graph",
            n=5,
        )

    for label, recs in results.items():
        items = [r["recommended_item"] for r in recs.to_pylist()]
        print(f"\n[{label}]  seed=['Coffee']")
        if items:
            for item in items:
                print(f"  - {item}")
        else:
            print("  (no recommendations)")


# ---------------------------------------------------------------------------
# Pattern 3 — Facet as entity prefix  (facet_mode="entity")
# ---------------------------------------------------------------------------

def run_facet_as_entity():
    print("\n" + "=" * 60)
    print("Pattern 3 — Facet as entity prefix  (cross-domain portable)")
    print("=" * 60)

    with Duwhal() as db:
        db.load_interactions(
            RAW,
            set_col="order_id",
            node_col="item",
            facet_cols=FACETS,
            facet_mode="entity",   # ← the only new argument
        )

        # Seed with a real item — graph routes through facet pseudo-nodes
        recs = db.recommend(["Pasta"], strategy="graph", n=8)
        print("\nSeed: ['Pasta']")
        print("Recommendations (graph, facet-enriched):")
        for row in recs.to_pylist():
            tag = "  [facet]" if row["recommended_item"].startswith("facet:") else ""
            print(f"  - {row['recommended_item']:45}  score={row['total_strength']:.4f}{tag}")

        # Seed with a facet — retrieve items associated with that context
        facet_seed = ["facet:region=EU|facet:day_period=night"]
        facet_recs = db.recommend(facet_seed, strategy="graph", n=5)
        print(f"\nSeed: {facet_seed}")
        print("Recommendations (items for EU / night context):")
        for row in facet_recs.to_pylist():
            if not row["recommended_item"].startswith("facet:"):
                print(f"  - {row['recommended_item']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Context Enrichment Without Losing Domain-Agnosticism")
    print("Demonstrating three facet patterns using the duwhal API.\n")
    run_composite_key()
    run_parallel_facets()
    run_facet_as_entity()
    print("\nDone.")
