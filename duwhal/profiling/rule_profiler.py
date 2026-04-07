"""
duwhal.profiling.rule_profiler
==============================
Cluster association rules in their natural metric space to surface latent
"recommendation personas" without requiring any hard-coded labels or
a fixed number of clusters.

Design principles
-----------------
* **No hard-coded labels**: cluster descriptions are derived entirely from
  data via the dominant-metric heuristic (highest absolute z-score from
  the global centroid).
* **Auto-k**: when ``n_clusters="auto"`` (the default), the best k is
  selected over ``[2, max_k]`` using the silhouette criterion.
* **Graceful degradation**: tiny rule sets (< 2 rules) fall back to a
  single cluster instead of raising.
* **Composability**: ``RuleProfiler`` wraps a ``pa.Table`` (the output of
  ``Duwhal.association_rules()``) and its ``recommend()`` method returns a
  ``pa.Table`` consistent with the rest of the Duwhal API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pyarrow as pa

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metrics produced by AssociationRules that form the clustering feature space.
METRIC_COLS: List[str] = [
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhang",
]

_CONVICTION_CAP = 1e6  # +inf conviction is replaced with this finite value

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "duwhal.profiling requires scikit-learn. "
            "Install it with:\n\n    pip install scikit-learn\n\n"
            "or add the [profiling] extra:\n\n    pip install duwhal[profiling]"
        )


def _cap_infinity(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values with a large finite sentinel and clip."""
    X = X.copy()
    X = np.where(np.isfinite(X), X, _CONVICTION_CAP)
    return np.clip(X, -_CONVICTION_CAP, _CONVICTION_CAP)


def _build_feature_matrix(rules_pd) -> np.ndarray:
    """Extract and sanitise the metric columns into a 2-D float array."""
    X = rules_pd[METRIC_COLS].values.astype(float)
    return _cap_infinity(X)


def _select_k(X_scaled: np.ndarray, max_k: int, random_state: int) -> int:
    """
    Return the k in ``[2, min(max_k, n_samples - 1)]`` that maximises the
    silhouette score. Falls back to k=1 for degenerate inputs.
    """
    n = X_scaled.shape[0]
    k_max = min(max_k, n - 1)
    if k_max < 2:
        return 1

    best_k, best_score = 2, -1.0
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X_scaled, labels))
        if score > best_score:
            best_score, best_k = score, k

    return best_k


def _dominant_metric(
    mask: np.ndarray,
    X_raw: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> str:
    """
    Return the name of the metric whose cluster-mean deviates most from the
    global mean (measured in standard-deviation units). This avoids any
    hard-coded ranking assumptions and lets the data speak for itself.
    """
    cluster_mean = X_raw[mask].mean(axis=0)
    safe_std = np.where(global_std > 0, global_std, 1.0)
    z_scores = np.abs((cluster_mean - global_mean) / safe_std)
    return METRIC_COLS[int(np.argmax(z_scores))]


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


@dataclass
class RuleCluster:
    """
    A fitted cluster of association rules sharing a common metric profile.

    Attributes
    ----------
    id : int
        Zero-based cluster index assigned by the clustering algorithm.
    dominant_metric : str
        The metric that distinguishes this cluster most from the overall
        average (highest absolute z-score). One of ``METRIC_COLS``.
    centroid : dict[str, float]
        Mean value of every metric for rules in this cluster.
    size : int
        Number of rules assigned to this cluster.
    label : str or None
        Optional human-friendly name, set by the caller via
        ``RuleProfiler.label()``. ``None`` until explicitly assigned.
    """

    id: int
    dominant_metric: str
    centroid: Dict[str, float]
    size: int
    label: Optional[str] = field(default=None)

    def __repr__(self) -> str:  # pragma: no cover
        name = self.label or f"Cluster-{self.id}"
        return (
            f"RuleCluster({name!r}, dominant={self.dominant_metric!r}, "
            f"size={self.size})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RuleProfiler:
    """
    Cluster association rules in metric-space to reveal latent recommendation
    personas.

    Parameters
    ----------
    rules : pa.Table
        Output of ``Duwhal.association_rules()``.  Must contain all columns
        in ``METRIC_COLS`` plus ``antecedents`` and ``consequents``.
    n_clusters : int or "auto"
        Number of clusters. ``"auto"`` (default) selects the best k via the
        silhouette score over ``[2, max_k]``.
    max_k : int
        Upper bound for the search when ``n_clusters="auto"`` (default 8).
    random_state : int
        Reproducibility seed for KMeans (default 0).

    Examples
    --------
    >>> with Duwhal() as db:
    ...     db.load_interactions(df, set_col="order_id", node_col="item_id")
    ...     db.association_rules(min_support=0.05, min_confidence=0.4)
    ...     profiler = db.profile_rules()          # fitted RuleProfiler
    ...     print(profiler.clusters)               # inspect what emerged
    ...     recs = profiler.recommend(["Pasta"])   # ranked by dominant metric
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
        self._clusters: List[RuleCluster] = []
        self._label_array: np.ndarray = np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "RuleProfiler":
        """Fit the profiler. Returns ``self`` for method chaining."""
        rules_pd = self._rules.to_pandas()
        X_raw = _build_feature_matrix(rules_pd)
        n_samples = len(rules_pd)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        if n_samples < 2:
            # Degenerate: single rule → trivial single cluster
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

        self._label_array = labels
        global_mean = X_raw.mean(axis=0)
        global_std = X_raw.std(axis=0)

        self._clusters = []
        for cid in sorted(set(labels.tolist())):
            mask = labels == cid
            cluster_mean = X_raw[mask].mean(axis=0)
            centroid = {m: round(float(cluster_mean[i]), 6) for i, m in enumerate(METRIC_COLS)}
            self._clusters.append(
                RuleCluster(
                    id=int(cid),
                    dominant_metric=_dominant_metric(mask, X_raw, global_mean, global_std),
                    centroid=centroid,
                    size=int(mask.sum()),
                )
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def clusters(self) -> List[RuleCluster]:
        """List of fitted :class:`RuleCluster` objects."""
        self._assert_fitted()
        return list(self._clusters)

    def label(self, mapping: Dict[int, str]) -> "RuleProfiler":
        """
        Attach human-friendly labels to cluster IDs.

        Parameters
        ----------
        mapping : dict[int, str]
            E.g. ``{0: "Staple", 1: "Discovery"}``.

        Returns
        -------
        RuleProfiler
            ``self``, so calls can be chained.
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
        Return a :class:`pa.Table` summarising each cluster's profile.

        Columns: ``cluster_id``, ``label``, ``dominant_metric``, ``size``,
        then one column per metric in ``METRIC_COLS`` (centroid value).
        """
        self._assert_fitted()
        rows = []
        for c in self._clusters:
            row: Dict = {
                "cluster_id": c.id,
                "label": c.label if c.label is not None else "",
                "dominant_metric": c.dominant_metric,
                "size": c.size,
            }
            row.update(c.centroid)
            rows.append(row)
        return pa.Table.from_pylist(rows)

    def rules_table(self) -> pa.Table:
        """
        Return the original rules table with two extra columns appended:
        ``cluster_id`` (int) and ``dominant_metric`` (str).
        """
        self._assert_fitted()
        id_to_dominant = {c.id: c.dominant_metric for c in self._clusters}
        ids = self._label_array.tolist()
        dominants = [id_to_dominant[i] for i in ids]
        result = self._rules
        result = result.append_column(
            "cluster_id", pa.array(ids, type=pa.int32())
        )
        result = result.append_column(
            "dominant_metric", pa.array(dominants, type=pa.string())
        )
        return result

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
        cluster: Optional[Union[int, str]] = None,
        n: int = 10,
    ) -> pa.Table:
        """
        Generate ranked recommendations from one or all clusters.

        Within each cluster, candidates are ranked by the cluster's own
        ``dominant_metric``, so the ranking criterion adapts to the data
        without any hard-coding.

        Parameters
        ----------
        seed_items : list[str]
            Items already present in the basket / context.
        cluster : int, str, or None
            Cluster ID (``int``) or label (``str``) to restrict results to.
            ``None`` (default) draws from all clusters (global dedup by
            highest score).
        n : int
            Maximum number of recommendations to return.

        Returns
        -------
        pa.Table
            Columns: ``item_id``, ``score``, ``cluster_id``,
            ``dominant_metric``, ``rule``.
        """
        self._assert_fitted()
        seed_set = set(seed_items)
        target_ids = self._resolve_cluster_filter(cluster)

        enriched = self.rules_table().to_pandas()

        antecedent_match = enriched["antecedents"].apply(
            lambda a: set(a.split("|")).issubset(seed_set)
        )
        not_in_seed = ~enriched["consequents"].isin(seed_set)

        if target_ids is not None:
            in_cluster = enriched["cluster_id"].isin(target_ids)
            filtered = enriched[antecedent_match & not_in_seed & in_cluster]
        else:
            filtered = enriched[antecedent_match & not_in_seed]

        if filtered.empty:
            return pa.Table.from_pylist([], schema=self._REC_SCHEMA)

        # Rank within each cluster by its dominant metric; then global dedup
        best: Dict[str, dict] = {}
        for cid, group in filtered.groupby("cluster_id"):
            dominant = group["dominant_metric"].iloc[0]
            sort_col = dominant if dominant in group.columns else "lift"
            for _, row in group.sort_values(sort_col, ascending=False).iterrows():
                item = str(row["consequents"])
                score = float(row[sort_col])
                if item not in best or score > best[item]["score"]:
                    best[item] = {
                        "item_id": item,
                        "score": score,
                        "cluster_id": int(cid),
                        "dominant_metric": dominant,
                        "rule": f"{row['antecedents']} \u2192 {row['consequents']}",
                    }

        ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:n]
        return pa.Table.from_pylist(ranked, schema=self._REC_SCHEMA)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "RuleProfiler has not been fitted yet. Call .fit() first."
            )

    def _resolve_cluster_filter(
        self, cluster: Optional[Union[int, str]]
    ) -> Optional[set]:
        if cluster is None:
            return None
        if isinstance(cluster, int):
            available = {c.id for c in self._clusters}
            if cluster not in available:
                raise ValueError(
                    f"Cluster ID {cluster!r} does not exist. "
                    f"Available: {sorted(available)}"
                )
            return {cluster}
        # string label
        matched = {c.id for c in self._clusters if c.label == cluster}
        if not matched:
            raise ValueError(
                f"No cluster with label {cluster!r}. "
                f"Assign labels with .label({{id: name}}) first."
            )
        return matched
