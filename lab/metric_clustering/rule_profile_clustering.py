"""
lab/rule_profile_clustering.py
==============================
Investigates two complementary approaches to recommendation profiling:

  1. Rule-Profile Approach
     ---------------------
     Treats each association rule as a point in a multi-metric space
     (support, confidence, lift, leverage, conviction, zhang).
     Rules are segmented into named "personas" using clustering, then
     recommendations are generated separately per persona, giving insight
     into *why* an item was recommended (safe, adventurous, staple, etc.).

  2. Item-Profile Approach
     ---------------------
     Aggregates rule-metrics *per item* (as consequent) to build an
     "item profile" vector. Items are then clustered by their profile,
     so you can identify "Gateway Items", "Symmetric Staples",
     "Niche Discoveries", etc., before personalization.

Run:
    uv run python lab/rule_profile_clustering.py
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from duwhal import Duwhal
from duwhal.datasets import generate_retail_transactions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_RULE_CLUSTERS = 3     # personas: "Staple", "Gateway", "Discovery"
N_ITEM_CLUSTERS = 3     # item archetypes
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mine_rules(db: Duwhal) -> pd.DataFrame:
    """Mine association rules and return a Pandas DataFrame with all metrics."""
    rules = db.association_rules(
        min_support=0.05,
        min_confidence=0.4,
        min_lift=1.0,
    ).to_pandas()
    return rules


METRIC_COLS = ["support", "confidence", "lift", "leverage", "conviction", "zhang"]


def _safe_clip_conviction(df: pd.DataFrame) -> pd.DataFrame:
    """conviction can be +inf; cap it to a large finite value for clustering."""
    df = df.copy()
    df["conviction"] = df["conviction"].replace(float("inf"), 1e6).clip(upper=1e6)
    return df


# ---------------------------------------------------------------------------
# 1. Rule-Profile Approach
# ---------------------------------------------------------------------------

def cluster_rules(rules: pd.DataFrame, n_clusters: int = N_RULE_CLUSTERS) -> pd.DataFrame:
    """
    Cluster rules in metric-space. Returns a copy of `rules` with an added
    'rule_cluster' column and a 'rule_persona' label.
    """
    rules = _safe_clip_conviction(rules)
    X = rules[METRIC_COLS].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
    rules = rules.copy()
    rules["rule_cluster"] = km.fit_predict(X_scaled)

    # ------------------------------------------------------------------
    # Label clusters by their avg lift (highest = Discovery, etc.)
    # ------------------------------------------------------------------
    cluster_lift = rules.groupby("rule_cluster")["lift"].mean().sort_values(ascending=False)
    persona_labels = ["Discovery", "Gateway", "Staple"]  # highest→lowest lift
    lift_to_persona = {cluster: persona_labels[i] for i, cluster in enumerate(cluster_lift.index)}
    rules["rule_persona"] = rules["rule_cluster"].map(lift_to_persona)

    return rules


def rule_profile_recommendations(
    rules_with_clusters: pd.DataFrame,
    seed_items: list[str],
    n: int = 5,
) -> pd.DataFrame:
    """
    For each rule persona, find the top-n recommendations for `seed_items`.
    Returns a DataFrame with columns: persona, item_id, score (lift), rule.
    """
    seed_set = set(seed_items)
    results = []

    for persona, group in rules_with_clusters.groupby("rule_persona"):
        matches = group[
            group["antecedents"].apply(lambda a: set(a.split("|")).issubset(seed_set))
            & ~group["consequents"].isin(seed_set)
        ].sort_values("lift", ascending=False).head(n)

        for _, row in matches.iterrows():
            results.append({
                "persona": persona,
                "item_id": row["consequents"],
                "lift": round(row["lift"], 3),
                "confidence": round(row["confidence"], 3),
                "zhang": round(row["zhang"], 3),
                "rule": f"{row['antecedents']} → {row['consequents']}",
            })

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["persona", "item_id", "lift", "confidence", "zhang", "rule"]
    )


# ---------------------------------------------------------------------------
# 2. Item-Profile Approach
# ---------------------------------------------------------------------------

def build_item_profiles(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rule metrics per consequent item to build an item profile.
    Stats: mean, max per metric.
    """
    rules = _safe_clip_conviction(rules)
    agg = rules.groupby("consequents")[METRIC_COLS].agg(["mean", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]
    # Also count how many rules fire for this item
    agg["rule_count"] = rules.groupby("consequents").size()
    return agg.reset_index().rename(columns={"consequents": "item_id"})


def cluster_items(item_profiles: pd.DataFrame, n_clusters: int = N_ITEM_CLUSTERS) -> pd.DataFrame:
    """
    Cluster items by their aggregated metric profiles.
    Returns the item_profiles DataFrame with 'item_cluster' and 'item_archetype'.
    """
    feature_cols = [c for c in item_profiles.columns if c not in ("item_id",)]
    X = item_profiles[feature_cols].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
    item_profiles = item_profiles.copy()
    item_profiles["item_cluster"] = km.fit_predict(X_scaled)

    # Label by avg mean_lift
    cluster_lift = item_profiles.groupby("item_cluster")["lift_mean"].mean().sort_values(ascending=False)
    archetype_labels = ["Niche Discovery", "Gateway Item", "Symmetric Staple"]
    cluster_to_arch = {c: archetype_labels[i] for i, c in enumerate(cluster_lift.index)}
    item_profiles["item_archetype"] = item_profiles["item_cluster"].map(cluster_to_arch)

    return item_profiles


def item_profile_recommendations(
    item_profiles: pd.DataFrame,
    seed_items: list[str],
    rules: pd.DataFrame,
    strategy: str = "Niche Discovery",
    n: int = 5,
) -> pd.DataFrame:
    """
    Recommend items from a specific archetype cluster.
    Filters rules where antecedents match seeds, then ranks by the
    archetype's key metric.
    """
    # Which items belong to the target archetype?
    archetype_items = set(
        item_profiles[item_profiles["item_archetype"] == strategy]["item_id"]
    )

    seed_set = set(seed_items)
    strategy_metric = {
        "Niche Discovery": "lift",
        "Gateway Item": "confidence",
        "Symmetric Staple": "zhang",
    }.get(strategy, "lift")

    matches = rules[
        rules["antecedents"].apply(lambda a: set(a.split("|")).issubset(seed_set))
        & rules["consequents"].isin(archetype_items)
        & ~rules["consequents"].isin(seed_set)
    ].sort_values(strategy_metric, ascending=False).head(n)

    return matches[["antecedents", "consequents", "support", "confidence", "lift", "zhang"]].copy()


# ---------------------------------------------------------------------------
# PCA Visualization helper (text-based, no matplotlib required)
# ---------------------------------------------------------------------------

def pca_summary(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> None:
    """Print a simple text PCA summary showing cluster centroids on PC1/PC2."""
    X = StandardScaler().fit_transform(df[feature_cols].values.astype(float))
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)
    df = df.copy()
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]
    print(f"\n  Explained variance by PC1/PC2: {pca.explained_variance_ratio_.round(3)}")
    summary = df.groupby(label_col)[["PC1", "PC2"]].mean().round(3)
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("DUWHAL — Rule & Item Profile Clustering Lab")
    print("=" * 65)

    df = generate_retail_transactions(seed=SEED)

    with Duwhal() as db:
        db.load_interactions(df, set_col="order_id", node_col="item_name")
        rules = mine_rules(db)

    print(f"\n[Data] {len(df)} transactions  |  {len(rules)} rules mined")
    print("\nTop 5 rules by lift:")
    print(
        rules[["antecedents", "consequents"] + METRIC_COLS]
        .sort_values("lift", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 1. RULE-PROFILE CLUSTERING
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("1. RULE-PROFILE CLUSTERING")
    print("─" * 65)

    rules_clustered = cluster_rules(rules)
    persona_counts = rules_clustered["rule_persona"].value_counts()
    print("\nRules per persona:")
    print(persona_counts.to_string())

    print("\nPersona centroid averages:")
    print(
        rules_clustered.groupby("rule_persona")[METRIC_COLS].mean().round(3).to_string()
    )

    seed = ["Pasta"]
    print(f"\nRule-profile recommendations for seed {seed}:")
    rp_recs = rule_profile_recommendations(rules_clustered, seed, n=3)
    if rp_recs.empty:
        print("  No rules matched the seed items.")
    else:
        print(rp_recs.to_string(index=False))

    feature_cols_rules = [c for c in rules_clustered.columns if c in METRIC_COLS]
    print("\nPCA centroid positions (Rule-space):")
    pca_summary(rules_clustered, feature_cols_rules, "rule_persona")

    # ------------------------------------------------------------------
    # 2. ITEM-PROFILE CLUSTERING
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("2. ITEM-PROFILE CLUSTERING")
    print("─" * 65)

    item_profiles = build_item_profiles(rules)
    item_profiles_clustered = cluster_items(item_profiles)

    print("\nItem archetypes:")
    print(item_profiles_clustered[["item_id", "item_archetype", "lift_mean", "confidence_mean", "zhang_mean"]]
          .sort_values(["item_archetype", "lift_mean"], ascending=[True, False])
          .to_string(index=False))

    print("\nArchetype centroid averages:")
    profile_feat_cols = [c for c in item_profiles.columns if c != "item_id"]
    print(
        item_profiles_clustered.groupby("item_archetype")[["lift_mean", "confidence_mean", "zhang_mean", "rule_count"]]
        .mean()
        .round(3)
        .to_string()
    )

    print("\n--- Item-profile recommendations by archetype ---")
    for archetype in ["Niche Discovery", "Gateway Item", "Symmetric Staple"]:
        recs = item_profile_recommendations(item_profiles_clustered, seed, rules, strategy=archetype, n=3)
        print(f"\n  [{archetype}] for seed {seed}:")
        if recs.empty:
            print("    No matches.")
        else:
            print(recs.to_string(index=False))

    print("\nPCA centroid positions (Item-profile space):")
    pca_summary(item_profiles_clustered, profile_feat_cols, "item_archetype")

    # ------------------------------------------------------------------
    # 3. SIDE-BY-SIDE COMPARISON (all seeds)
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("3. SIDE-BY-SIDE: Rule-Profile vs Item-Profile")
    print("─" * 65)

    all_items = rules["antecedents"].str.split("|").explode().unique().tolist()
    print(f"\nTesting {len(all_items)} unique seed items...\n")

    comparison = []
    for item in all_items:
        rp = rule_profile_recommendations(rules_clustered, [item], n=1)
        ip = item_profile_recommendations(item_profiles_clustered, [item], rules, strategy="Niche Discovery", n=1)

        rp_top = rp["item_id"].iloc[0] if not rp.empty else "—"
        rp_persona = rp["persona"].iloc[0] if not rp.empty else "—"
        ip_top = ip["consequents"].iloc[0] if not ip.empty else "—"

        comparison.append({
            "seed": item,
            "rule_profile_rec": rp_top,
            "rule_persona": rp_persona,
            "item_profile_rec (Niche)": ip_top,
        })

    print(pd.DataFrame(comparison).to_string(index=False))
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
