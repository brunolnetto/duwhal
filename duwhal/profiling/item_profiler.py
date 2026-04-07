"""
duwhal.profiling.item_profiler
==============================
Cluster items (as rule consequents) by aggregating their rule-metric
fingerprints, exposing stable "archetypes" without hard-coded labels.

The item profile of a consequent is the vector of (mean, max) statistics
across all rules that fire it, plus a ``rule_count`` feature.  KMeans then
groups items by shared metric behaviour; the ``dominant_metric`` of each
cluster is the feature with the highest absolute z-score relative to the
global average — fully data-driven.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
import pyarrow as pa

from .rule_profiler import (
    _CONVICTION_CAP,
    _SKLEARN_AVAILABLE,
    METRIC_COLS,
    _cap_infinity,
    _require_sklearn,
    _select_k,
)

if _SKLEARN_AVAILABLE:
    from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_ITEM_FEATURE_NAMES: List[str] = (
    [f"{m}_mean" for m in METRIC_COLS]
    + [f"{m}_max" for m in METRIC_COLS]
    + ["rule_count"]
)


def _build_item_profiles(rules_pd) -> "pd.DataFrame":
    """
    Aggregate rule metrics per consequent to build a per-item feature frame.

    Each row represents one unique consequent; features are mean and max of
    every metric across all rules that recommend it, plus a rule_count column.
    """
    import pandas as pd

    rules_pd = rules_pd.copy()
    rules_pd["conviction"] = rules_pd["conviction"].replace(float("inf"), _CONVICTION_CAP)
    rules_pd["conviction"] = rules_pd["conviction"].clip(upper=_CONVICTION_CAP)

    grouped = rules_pd.groupby("consequents")[METRIC_COLS]
    agg_mean = grouped.mean().add_suffix("_mean")
    agg_max = grouped.max().add_suffix("_max")
    rule_count = grouped.size().rename("rule_count")

    profiles = pd.concat([agg_mean, agg_max, rule_count], axis=1).reset_index()
    profiles = profiles.rename(columns={"consequents": "item_id"})
    return profiles


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


@dataclass
class ItemCluster:
    """
    A cluster of items sharing a common rule-metric fingerprint.

    Attributes
    ----------
    id : int
        Zero-based cluster index.
    dominant_metric : str
        The aggregated feature (e.g. ``"lift_mean"``) with the greatest
        absolute z-score relative to the overall item population.
    centroid : dict[str, float]
        Mean value of every item-feature for this cluster.
    size : int
        Number of items in this cluster.
    label : str or None
        Optional caller-supplied name; ``None`` until set via
        ``ItemProfiler.label()``.
    members : list[str]
        Item IDs assigned to this cluster.
    """

    id: int
    dominant_metric: str
    centroid: Dict[str, float]
    size: int
    label: Optional[str] = field(default=None)
    members: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        name = self.label or f"Cluster-{self.id}"
        return (
            f"ItemCluster({name!r}, dominant={self.dominant_metric!r}, "
            f"size={self.size})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ItemProfiler:
    """
    Cluster items by their rule-metric fingerprints to reveal stable archetypes.

    Parameters
    ----------
    rules : pa.Table
        Output of ``Duwhal.association_rules()``.
    n_clusters : int or "auto"
        Number of clusters. ``"auto"`` (default) selects the best k via the
        silhouette score.
    max_k : int
        Upper bound for k search when ``n_clusters="auto"`` (default 8).
    random_state : int
        Reproducibility seed (default 0).

    Examples
    --------
    >>> with Duwhal() as db:
    ...     db.load_interactions(df, set_col="order_id", node_col="item_id")
    ...     db.association_rules(min_support=0.05, min_confidence=0.4)
    ...     profiler = db.profile_items()
    ...     print(profiler.clusters)
    ...     recs = profiler.recommend(["Pasta"], archetype=0)
    """

    def __init__(
        self,
        rules: pa.Table,
        n_clusters: Union[int, Literal["auto"]] = "auto",
        max_k: int = 8,
        random_state: int = 0,
    ) -> None:
        _require_sklearn()
        if rules.num_rows == 0:
            raise ValueError(
                "Cannot profile an empty rules table. "
                "Lower min_support or min_confidence."
            )
        self._rules = rules
        self._n_clusters = n_clusters
        self._max_k = max(2, int(max_k))
        self._random_state = int(random_state)
        self._fitted = False
        self._clusters: List[ItemCluster] = []
        self._item_profiles = None  # pd.DataFrame after fit

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "ItemProfiler":
        """Fit the profiler. Returns ``self`` for method chaining."""
        from sklearn.preprocessing import StandardScaler

        rules_pd = self._rules.to_pandas()
        profiles = _build_item_profiles(rules_pd)

        feature_cols = [c for c in profiles.columns if c != "item_id"]
        X_raw = profiles[feature_cols].values.astype(float)
        X_raw = _cap_infinity(X_raw)

        n_samples = len(profiles)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        if n_samples < 2:
            k = 1
        elif self._n_clusters == "auto":
            k = _select_k(X_scaled, self._max_k, self._random_state)
        else:
            k = min(int(self._n_clusters), n_samples)

        if k == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            km = KMeans(n_clusters=k, random_state=self._random_state, n_init="auto")
            labels = km.fit_predict(X_scaled)

        profiles = profiles.copy()
        profiles["_cluster_id"] = labels

        global_mean = X_raw.mean(axis=0)
        global_std = X_raw.std(axis=0)

        self._clusters = []
        for cid in sorted(set(labels.tolist())):
            mask = labels == cid
            cluster_mean = X_raw[mask].mean(axis=0)
            centroid = {
                feat: round(float(cluster_mean[i]), 6)
                for i, feat in enumerate(feature_cols)
            }
            # Dominant metric: highest absolute z-score in item feature space
            safe_std = np.where(global_std > 0, global_std, 1.0)
            z_scores = np.abs((cluster_mean - global_mean) / safe_std)
            dominant_name = feature_cols[int(np.argmax(z_scores))]
            members = profiles.loc[mask, "item_id"].tolist()
            self._clusters.append(
                ItemCluster(
                    id=int(cid),
                    dominant_metric=dominant_name,
                    centroid=centroid,
                    size=int(mask.sum()),
                    members=members,
                )
            )

        self._item_profiles = profiles
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def clusters(self) -> List[ItemCluster]:
        """List of fitted :class:`ItemCluster` objects."""
        self._assert_fitted()
        return list(self._clusters)

    def label(self, mapping: Dict[int, str]) -> "ItemProfiler":
        """
        Attach human-friendly labels to cluster IDs. Returns ``self``.

        Parameters
        ----------
        mapping : dict[int, str]
            E.g. ``{0: "Gateway", 1: "Niche"}``.
        """
        self._assert_fitted()
        id_map = {c.id: c for c in self._clusters}
        for cid, name in mapping.items():
            if cid in id_map:
                id_map[cid].label = str(name)
            else:
                warnings.warn(
                    f"Cluster ID {cid!r} does not exist. "
                    f"Available IDs: {sorted(id_map)}",
                    stacklevel=2,
                )
        return self

    def summary(self) -> pa.Table:
        """
        Return a :class:`pa.Table` summarising each cluster's item profile.

        Columns: ``cluster_id``, ``label``, ``dominant_metric``, ``size``,
        ``members`` (list<str>), then one column per aggregated feature.
        """
        self._assert_fitted()
        rows = []
        for c in self._clusters:
            row: Dict = {
                "cluster_id": c.id,
                "label": c.label if c.label is not None else "",
                "dominant_metric": c.dominant_metric,
                "size": c.size,
                "members": c.members,
            }
            row.update(c.centroid)
            rows.append(row)
        return pa.Table.from_pylist(rows)

    def cluster_of(self, item_id: str) -> Optional[ItemCluster]:
        """Return the :class:`ItemCluster` that ``item_id`` belongs to, or ``None``."""
        self._assert_fitted()
        for c in self._clusters:
            if item_id in c.members:
                return c
        return None

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    _REC_SCHEMA = pa.schema(
        [
            ("item_id", pa.string()),
            ("score", pa.float64()),
            ("cluster_id", pa.int32()),
            ("dominant_metric", pa.string()),
            ("rule", pa.string()),
        ]
    )

    def recommend(
        self,
        seed_items: List[str],
        archetype: Optional[Union[int, str]] = None,
        n: int = 10,
    ) -> pa.Table:
        """
        Generate recommendations filtered by item archetype.

        Candidates must be consequents of rules whose antecedents are a
        subset of ``seed_items``.  Within each archetype, items are ranked
        by the cluster's ``dominant_metric`` so the scoring adapts to the
        data.

        Parameters
        ----------
        seed_items : list[str]
            Items already in context.
        archetype : int, str, or None
            Cluster ID (``int``) or label (``str``) to restrict to.
            ``None`` = all archetypes.
        n : int
            Maximum recommendations to return.

        Returns
        -------
        pa.Table
            Columns: ``item_id``, ``score``, ``cluster_id``,
            ``dominant_metric``, ``rule``.
        """
        self._assert_fitted()
        seed_set = set(seed_items)
        target_ids = self._resolve_archetype(archetype)

        rules_pd = self._rules.to_pandas()
        profiles = self._item_profiles

        # Map each item to its cluster
        item_to_cluster = {
            row["item_id"]: int(row["_cluster_id"])
            for _, row in profiles.iterrows()
        }
        cluster_to_dominant = {c.id: c.dominant_metric for c in self._clusters}

        antecedent_match = rules_pd["antecedents"].apply(
            lambda a: set(a.split("|")).issubset(seed_set)
        )
        not_in_seed = ~rules_pd["consequents"].isin(seed_set)
        filtered = rules_pd[antecedent_match & not_in_seed].copy()

        if filtered.empty:
            return pa.Table.from_pylist([], schema=self._REC_SCHEMA)

        filtered["_cid"] = filtered["consequents"].map(item_to_cluster)

        if target_ids is not None:
            filtered = filtered[filtered["_cid"].isin(target_ids)]

        if filtered.empty:
            return pa.Table.from_pylist([], schema=self._REC_SCHEMA)

        best: Dict[str, dict] = {}
        for cid, group in filtered.groupby("_cid"):
            # Use the metric name from the item feature space; strip
            # aggregation suffix to get the raw metric for ranking in rules
            raw_dom = cluster_to_dominant.get(int(cid), "lift_mean")
            # Fall back to the base metric (e.g. "lift_mean" → "lift")
            sort_col = raw_dom.replace("_mean", "").replace("_max", "")
            if sort_col not in group.columns:
                sort_col = "lift"
            dominant_display = cluster_to_dominant.get(int(cid), "lift_mean")
            for _, row in group.sort_values(sort_col, ascending=False).iterrows():
                item = str(row["consequents"])
                score = float(row[sort_col])
                if item not in best or score > best[item]["score"]:
                    best[item] = {
                        "item_id": item,
                        "score": score,
                        "cluster_id": int(cid),
                        "dominant_metric": dominant_display,
                        "rule": f"{row['antecedents']} \u2192 {row['consequents']}",
                    }

        ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:n]
        return (
            pa.Table.from_pylist(ranked, schema=self._REC_SCHEMA)
            if ranked
            else pa.Table.from_pylist([], schema=self._REC_SCHEMA)
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "ItemProfiler has not been fitted yet. Call .fit() first."
            )

    def _resolve_archetype(self, archetype: Optional[Union[int, str]]) -> Optional[set]:
        if archetype is None:
            return None
        if isinstance(archetype, int):
            available = {c.id for c in self._clusters}
            if archetype not in available:
                raise ValueError(
                    f"Cluster ID {archetype!r} does not exist. "
                    f"Available: {sorted(available)}"
                )
            return {archetype}
        # string label
        matched = {c.id for c in self._clusters if c.label == archetype}
        if not matched:
            raise ValueError(
                f"No archetype with label {archetype!r}. "
                "Assign labels with .label({id: name}) first."
            )
        return matched
