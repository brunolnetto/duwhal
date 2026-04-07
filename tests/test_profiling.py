"""
tests/test_profiling.py
=======================
Tests for duwhal.profiling — RuleProfiler and ItemProfiler.

Coverage goals
--------------
* Auto-k selection and fixed-k
* Cluster introspection (.clusters, .summary(), .rules_table())
* User labelling via .label() + warning on unknown id
* recommend() with no filter, cluster-id filter, label filter
* recommend() degenerate cases (no seed match, empty result)
* ItemProfiler analogues of all the above
* Integration via Duwhal.profile_rules() / Duwhal.profile_items()
* RuntimeError guards (unfitted, missing rules)
* ValueError guards (bad cluster id, bad label)
* Degenerate rule tables (1 rule, 2 rules)
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

from duwhal import Duwhal
from duwhal.datasets import generate_retail_transactions
from duwhal.profiling import ItemProfiler, RuleProfiler
from duwhal.profiling.rule_profiler import METRIC_COLS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retail_rules() -> pa.Table:
    """A rich rule table mined from the retail dataset."""
    df = generate_retail_transactions(seed=0)
    with Duwhal() as db:
        db.load_interactions(df, set_col="order_id", node_col="item_name")
        return db.association_rules(min_support=0.05, min_confidence=0.3)


@pytest.fixture(scope="module")
def fitted_rule_profiler(retail_rules: pa.Table) -> RuleProfiler:
    return RuleProfiler(retail_rules, n_clusters="auto", random_state=0).fit()


@pytest.fixture(scope="module")
def fitted_item_profiler(retail_rules: pa.Table) -> ItemProfiler:
    return ItemProfiler(retail_rules, n_clusters="auto", random_state=0).fit()


@pytest.fixture(scope="module")
def tiny_rules() -> pa.Table:
    """A minimal 2-rule table to test degenerate behaviour."""
    return pa.Table.from_pylist([
        {
            "antecedents": "A",
            "consequents": "B",
            "support": 0.5,
            "confidence": 1.0,
            "lift": 2.0,
            "leverage": 0.25,
            "conviction": float("inf"),
            "zhang": 1.0,
        },
        {
            "antecedents": "B",
            "consequents": "A",
            "support": 0.5,
            "confidence": 1.0,
            "lift": 2.0,
            "leverage": 0.25,
            "conviction": float("inf"),
            "zhang": 1.0,
        },
    ])


@pytest.fixture(scope="module")
def single_rule() -> pa.Table:
    """One-row rule table to test the degenerate single-cluster path."""
    return pa.Table.from_pylist([
        {
            "antecedents": "X",
            "consequents": "Y",
            "support": 0.3,
            "confidence": 0.9,
            "lift": 3.0,
            "leverage": 0.1,
            "conviction": 5.0,
            "zhang": 0.8,
        }
    ])


# ===========================================================================
# RuleProfiler — core
# ===========================================================================

class TestRuleProfilerCore:

    def test_fit_returns_self(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0)
        assert profiler.fit() is profiler

    def test_clusters_type_and_length(self, fitted_rule_profiler):
        clusters = fitted_rule_profiler.clusters
        assert isinstance(clusters, list)
        assert len(clusters) >= 1

    def test_cluster_sizes_sum_to_total(self, retail_rules, fitted_rule_profiler):
        total = sum(c.size for c in fitted_rule_profiler.clusters)
        assert total == retail_rules.num_rows

    def test_cluster_ids_are_unique(self, fitted_rule_profiler):
        ids = [c.id for c in fitted_rule_profiler.clusters]
        assert len(ids) == len(set(ids))

    def test_dominant_metric_is_valid(self, fitted_rule_profiler):
        for c in fitted_rule_profiler.clusters:
            assert c.dominant_metric in METRIC_COLS

    def test_centroid_has_all_metrics(self, fitted_rule_profiler):
        for c in fitted_rule_profiler.clusters:
            assert set(c.centroid.keys()) == set(METRIC_COLS)

    def test_label_is_none_by_default(self, fitted_rule_profiler):
        for c in fitted_rule_profiler.clusters:
            assert c.label is None

    def test_fixed_k(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        assert len(profiler.clusters) == 2

    def test_fixed_k_capped_at_n_samples(self, tiny_rules):
        # k=100 should be silently reduced to ≤ n_samples
        profiler = RuleProfiler(tiny_rules, n_clusters=100, random_state=0).fit()
        assert len(profiler.clusters) <= tiny_rules.num_rows

    def test_single_rule_yields_one_cluster(self, single_rule):
        profiler = RuleProfiler(single_rule, n_clusters="auto", random_state=0).fit()
        assert len(profiler.clusters) == 1
        assert profiler.clusters[0].size == 1

    def test_empty_rules_raises(self):
        empty = pa.Table.from_pylist([], schema=pa.schema(
            [(m, pa.float64()) for m in METRIC_COLS]
            + [("antecedents", pa.string()), ("consequents", pa.string())]
        ))
        with pytest.raises(ValueError, match="empty"):
            RuleProfiler(empty)


# ===========================================================================
# RuleProfiler — inspection API
# ===========================================================================

class TestRuleProfilerInspection:

    def test_summary_shape(self, fitted_rule_profiler):
        summary = fitted_rule_profiler.summary()
        assert isinstance(summary, pa.Table)
        assert summary.num_rows == len(fitted_rule_profiler.clusters)

    def test_summary_columns(self, fitted_rule_profiler):
        cols = fitted_rule_profiler.summary().column_names
        assert "cluster_id" in cols
        assert "dominant_metric" in cols
        assert "size" in cols
        for m in METRIC_COLS:
            assert m in cols

    def test_rules_table_has_extra_columns(self, fitted_rule_profiler, retail_rules):
        rt = fitted_rule_profiler.rules_table()
        assert "cluster_id" in rt.column_names
        assert "dominant_metric" in rt.column_names
        assert rt.num_rows == retail_rules.num_rows

    def test_rules_table_cluster_ids_match_clusters(self, fitted_rule_profiler):
        rt = fitted_rule_profiler.rules_table().to_pandas()
        valid_ids = {c.id for c in fitted_rule_profiler.clusters}
        assert set(rt["cluster_id"].unique()).issubset(valid_ids)

    def test_unfitted_raises_on_clusters(self, retail_rules):
        p = RuleProfiler(retail_rules, n_clusters=2)
        with pytest.raises(RuntimeError, match="fitted"):
            _ = p.clusters

    def test_unfitted_raises_on_summary(self, retail_rules):
        p = RuleProfiler(retail_rules, n_clusters=2)
        with pytest.raises(RuntimeError, match="fitted"):
            p.summary()

    def test_unfitted_raises_on_rules_table(self, retail_rules):
        p = RuleProfiler(retail_rules, n_clusters=2)
        with pytest.raises(RuntimeError, match="fitted"):
            p.rules_table()


# ===========================================================================
# RuleProfiler — labelling
# ===========================================================================

class TestRuleProfilerLabelling:

    def test_label_sets_name(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        cid = profiler.clusters[0].id
        profiler.label({cid: "MyPersona"})
        assert profiler.clusters[0].label == "MyPersona"

    def test_label_returns_self(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        assert profiler.label({}) is profiler

    def test_label_unknown_id_warns(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profiler.label({999: "Ghost"})
        assert any("999" in str(w.message) for w in caught)


# ===========================================================================
# RuleProfiler — recommend
# ===========================================================================

class TestRuleProfilerRecommend:

    def test_recommend_returns_arrow_table(self, fitted_rule_profiler):
        recs = fitted_rule_profiler.recommend(["Pasta"])
        assert isinstance(recs, pa.Table)

    def test_recommend_columns(self, fitted_rule_profiler):
        recs = fitted_rule_profiler.recommend(["Pasta"])
        expected = {"item_id", "score", "cluster_id", "dominant_metric", "rule"}
        assert expected.issubset(set(recs.column_names))

    def test_recommend_excludes_seed(self, fitted_rule_profiler):
        recs = fitted_rule_profiler.recommend(["Pasta"])
        items = recs.column("item_id").to_pylist()
        assert "Pasta" not in items

    def test_recommend_respects_n(self, fitted_rule_profiler):
        recs = fitted_rule_profiler.recommend(["Pasta"], n=2)
        assert recs.num_rows <= 2

    def test_recommend_no_seed_match_returns_empty(self, fitted_rule_profiler):
        recs = fitted_rule_profiler.recommend(["__nonexistent__"])
        assert recs.num_rows == 0

    def test_recommend_cluster_int_filter(self, fitted_rule_profiler):
        cid = fitted_rule_profiler.clusters[0].id
        recs = fitted_rule_profiler.recommend(["Pasta"], cluster=cid)
        if recs.num_rows > 0:
            assert all(c == cid for c in recs.column("cluster_id").to_pylist())

    def test_recommend_cluster_label_filter(self, retail_rules):
        profiler = RuleProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        cid = profiler.clusters[0].id
        profiler.label({cid: "TestLabel"})
        recs = profiler.recommend(["Pasta"], cluster="TestLabel")
        if recs.num_rows > 0:
            assert all(c == cid for c in recs.column("cluster_id").to_pylist())

    def test_recommend_bad_cluster_id_raises(self, fitted_rule_profiler):
        with pytest.raises(ValueError, match="does not exist"):
            fitted_rule_profiler.recommend(["Pasta"], cluster=9999)

    def test_recommend_bad_label_raises(self, fitted_rule_profiler):
        with pytest.raises(ValueError, match="label"):
            fitted_rule_profiler.recommend(["Pasta"], cluster="__unknown__")


# ===========================================================================
# ItemProfiler — core
# ===========================================================================

class TestItemProfilerCore:

    def test_fit_returns_self(self, retail_rules):
        profiler = ItemProfiler(retail_rules, n_clusters=2, random_state=0)
        assert profiler.fit() is profiler

    def test_clusters_type(self, fitted_item_profiler):
        assert isinstance(fitted_item_profiler.clusters, list)
        assert len(fitted_item_profiler.clusters) >= 1

    def test_members_cover_all_consequents(self, retail_rules, fitted_item_profiler):
        all_consequents = set(retail_rules.to_pandas()["consequents"].unique())
        all_members: set = set()
        for c in fitted_item_profiler.clusters:
            all_members.update(c.members)
        assert all_consequents == all_members

    def test_dominant_metric_is_item_feature(self, fitted_item_profiler):
        from duwhal.profiling.item_profiler import _ITEM_FEATURE_NAMES
        for c in fitted_item_profiler.clusters:
            assert c.dominant_metric in _ITEM_FEATURE_NAMES

    def test_centroid_has_item_features(self, fitted_item_profiler):
        from duwhal.profiling.item_profiler import _ITEM_FEATURE_NAMES
        for c in fitted_item_profiler.clusters:
            assert set(c.centroid.keys()).issubset(set(_ITEM_FEATURE_NAMES))

    def test_single_rule_yields_one_cluster(self, single_rule):
        profiler = ItemProfiler(single_rule, n_clusters="auto", random_state=0).fit()
        assert len(profiler.clusters) == 1

    def test_empty_rules_raises(self):
        empty = pa.Table.from_pylist([], schema=pa.schema(
            [(m, pa.float64()) for m in METRIC_COLS]
            + [("antecedents", pa.string()), ("consequents", pa.string())]
        ))
        with pytest.raises(ValueError, match="empty"):
            ItemProfiler(empty)


# ===========================================================================
# ItemProfiler — inspection API
# ===========================================================================

class TestItemProfilerInspection:

    def test_summary_shape(self, fitted_item_profiler):
        summary = fitted_item_profiler.summary()
        assert isinstance(summary, pa.Table)
        assert summary.num_rows == len(fitted_item_profiler.clusters)

    def test_summary_has_members_column(self, fitted_item_profiler):
        assert "members" in fitted_item_profiler.summary().column_names

    def test_cluster_of_known_item(self, retail_rules, fitted_item_profiler):
        item = retail_rules.to_pandas()["consequents"].iloc[0]
        cluster = fitted_item_profiler.cluster_of(item)
        assert cluster is not None
        assert item in cluster.members

    def test_cluster_of_unknown_item_returns_none(self, fitted_item_profiler):
        assert fitted_item_profiler.cluster_of("__nonexistent__") is None

    def test_unfitted_raises_on_clusters(self, retail_rules):
        p = ItemProfiler(retail_rules, n_clusters=2)
        with pytest.raises(RuntimeError, match="fitted"):
            _ = p.clusters


# ===========================================================================
# ItemProfiler — labelling
# ===========================================================================

class TestItemProfilerLabelling:

    def test_label_sets_name(self, retail_rules):
        profiler = ItemProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        cid = profiler.clusters[0].id
        profiler.label({cid: "Staple"})
        assert profiler.clusters[0].label == "Staple"

    def test_label_returns_self(self, retail_rules):
        profiler = ItemProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        assert profiler.label({}) is profiler

    def test_label_unknown_id_warns(self, retail_rules):
        profiler = ItemProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profiler.label({999: "Ghost"})
        assert any("999" in str(w.message) for w in caught)


# ===========================================================================
# ItemProfiler — recommend
# ===========================================================================

class TestItemProfilerRecommend:

    def test_recommend_returns_arrow_table(self, fitted_item_profiler):
        recs = fitted_item_profiler.recommend(["Pasta"])
        assert isinstance(recs, pa.Table)

    def test_recommend_columns(self, fitted_item_profiler):
        recs = fitted_item_profiler.recommend(["Pasta"])
        expected = {"item_id", "score", "cluster_id", "dominant_metric", "rule"}
        assert expected.issubset(set(recs.column_names))

    def test_recommend_excludes_seed(self, fitted_item_profiler):
        recs = fitted_item_profiler.recommend(["Pasta"])
        items = recs.column("item_id").to_pylist()
        assert "Pasta" not in items

    def test_recommend_respects_n(self, fitted_item_profiler):
        recs = fitted_item_profiler.recommend(["Pasta"], n=1)
        assert recs.num_rows <= 1

    def test_recommend_no_seed_match_returns_empty(self, fitted_item_profiler):
        recs = fitted_item_profiler.recommend(["__nonexistent__"])
        assert recs.num_rows == 0

    def test_recommend_archetype_int_filter(self, fitted_item_profiler):
        cid = fitted_item_profiler.clusters[0].id
        recs = fitted_item_profiler.recommend(["Pasta"], archetype=cid)
        if recs.num_rows > 0:
            assert all(c == cid for c in recs.column("cluster_id").to_pylist())

    def test_recommend_archetype_label_filter(self, retail_rules):
        profiler = ItemProfiler(retail_rules, n_clusters=2, random_state=0).fit()
        cid = profiler.clusters[0].id
        profiler.label({cid: "Niche"})
        recs = profiler.recommend(["Pasta"], archetype="Niche")
        if recs.num_rows > 0:
            assert all(c == cid for c in recs.column("cluster_id").to_pylist())

    def test_recommend_bad_archetype_id_raises(self, fitted_item_profiler):
        with pytest.raises(ValueError, match="does not exist"):
            fitted_item_profiler.recommend(["Pasta"], archetype=9999)

    def test_recommend_bad_label_raises(self, fitted_item_profiler):
        with pytest.raises(ValueError, match="label"):
            fitted_item_profiler.recommend(["Pasta"], archetype="__unknown__")


# ===========================================================================
# Integration — Duwhal.profile_rules() / profile_items()
# ===========================================================================

class TestDuwhalIntegration:

    @pytest.fixture(scope="class")
    def db_with_rules(self):
        df = generate_retail_transactions(seed=0)
        db = Duwhal()
        db.__enter__()
        db.load_interactions(df, set_col="order_id", node_col="item_name")
        db.association_rules(min_support=0.05, min_confidence=0.3)
        yield db
        db.__exit__(None, None, None)

    def test_profile_rules_returns_rule_profiler(self, db_with_rules):
        profiler = db_with_rules.profile_rules(random_state=0)
        assert isinstance(profiler, RuleProfiler)
        assert profiler._fitted

    def test_profile_items_returns_item_profiler(self, db_with_rules):
        profiler = db_with_rules.profile_items(random_state=0)
        assert isinstance(profiler, ItemProfiler)
        assert profiler._fitted

    def test_profile_rules_without_prior_rules_raises(self):
        with Duwhal() as db:
            with pytest.raises(RuntimeError, match="association_rules"):
                db.profile_rules()

    def test_profile_items_without_prior_rules_raises(self):
        with Duwhal() as db:
            with pytest.raises(RuntimeError, match="association_rules"):
                db.profile_items()

    def test_full_pipeline(self, db_with_rules):
        """End-to-end: mine rules → profile → recommend."""
        rule_profiler = db_with_rules.profile_rules(n_clusters=2, random_state=0)
        item_profiler = db_with_rules.profile_items(n_clusters=2, random_state=0)

        recs_rules = rule_profiler.recommend(["Pasta"], n=3)
        recs_items = item_profiler.recommend(["Pasta"], n=3)

        assert isinstance(recs_rules, pa.Table)
        assert isinstance(recs_items, pa.Table)
        # Results must not contain the seed
        assert "Pasta" not in recs_rules.column("item_id").to_pylist()
        assert "Pasta" not in recs_items.column("item_id").to_pylist()
