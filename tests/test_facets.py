"""Tests for duwhal.core.facets and the facet-aware Duwhal API."""

import pandas as pd
import pyarrow as pa
import pytest

from duwhal import Duwhal
from duwhal.core.facets import (
    build_composite_key,
    build_facet_entities,
    merge_recommendation_tables,
    split_by_facet,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    return pd.DataFrame(
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


# ---------------------------------------------------------------------------
# build_composite_key
# ---------------------------------------------------------------------------

class TestBuildCompositeKey:

    def test_output_column_added(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region"], out_col="ctx")
        assert "ctx" in out.columns

    def test_default_out_col(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region"])
        assert "_context_key" in out.columns

    def test_key_contains_set_col_value(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region"], out_col="ctx")
        first = out["ctx"].iloc[0]
        assert "1" in first
        assert "EU" in first

    def test_key_with_multiple_facets(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region", "day_period"], out_col="ctx")
        first = out["ctx"].iloc[0]
        assert "EU" in first and "night" in first

    def test_original_columns_preserved(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region"], out_col="ctx")
        assert "order_id" in out.columns and "region" in out.columns
        assert len(out) == len(raw_df)

    def test_sep_propagated(self, raw_df):
        out = build_composite_key(raw_df, "order_id", ["region"], sep="##", out_col="ctx")
        assert "##" in out["ctx"].iloc[0]

    def test_missing_column_raises(self, raw_df):
        with pytest.raises(ValueError, match="missing columns"):
            build_composite_key(raw_df, "order_id", ["nonexistent"])

    def test_missing_set_col_raises(self, raw_df):
        with pytest.raises(ValueError, match="missing columns"):
            build_composite_key(raw_df, "bad_col", ["region"])


# ---------------------------------------------------------------------------
# build_facet_entities
# ---------------------------------------------------------------------------

class TestBuildFacetEntities:

    def test_returns_dataframe(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        assert isinstance(out, pd.DataFrame)

    def test_columns_are_set_col_and_node_col(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        assert list(out.columns) == ["order_id", "item"]

    def test_original_item_rows_present(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        assert set(raw_df["item"]) <= set(out["item"])

    def test_facet_entity_rows_present(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        assert any(out["item"].str.startswith("facet:"))

    def test_facet_label_format(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        labels = out[out["item"].str.startswith("facet:")]["item"].unique()
        assert any("region=EU" in lbl for lbl in labels)

    def test_multiple_facet_cols_joined(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region", "day_period"])
        labels = out[out["item"].str.startswith("facet:")]["item"].unique()
        assert any("region=" in lbl and "day_period=" in lbl for lbl in labels)

    def test_facet_rows_deduplicated_per_context(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"])
        order1_facets = out[(out["order_id"] == 1) & out["item"].str.startswith("facet:")]
        assert len(order1_facets) == 1

    def test_custom_prefix(self, raw_df):
        out = build_facet_entities(raw_df, "order_id", "item", ["region"], prefix="ctx")
        assert any(out["item"].str.startswith("ctx:"))

    def test_missing_facet_col_raises(self, raw_df):
        with pytest.raises(ValueError, match="missing columns"):
            build_facet_entities(raw_df, "order_id", "item", ["nonexistent"])


# ---------------------------------------------------------------------------
# split_by_facet
# ---------------------------------------------------------------------------

class TestSplitByFacet:

    def test_returns_dict(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", "region")
        assert isinstance(slices, dict)

    def test_single_facet_keys_use_col_eq_val_format(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", "region")
        assert set(slices.keys()) == {"region=EU", "region=BR"}

    def test_multi_facet_keys_are_compound(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", ["region", "day_period"])
        # All keys should encode both dimensions
        assert all("|" in k for k in slices.keys())

    def test_each_slice_has_correct_columns(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", "region")
        for df in slices.values():
            assert "order_id" in df.columns and "item" in df.columns

    def test_slices_contain_correct_rows(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", "region")
        assert "Pasta" in set(slices["region=EU"]["item"])
        assert "Coffee" in set(slices["region=BR"]["item"])
        assert "Coffee" not in set(slices["region=EU"]["item"])

    def test_extra_cols_retained(self, raw_df):
        slices = split_by_facet(raw_df, "order_id", "item", "region", extra_cols=["day_period"])
        for df in slices.values():
            assert "day_period" in df.columns

    def test_missing_facet_col_raises(self, raw_df):
        with pytest.raises(ValueError, match="missing columns"):
            split_by_facet(raw_df, "order_id", "item", "nonexistent")


# ---------------------------------------------------------------------------
# merge_recommendation_tables
# ---------------------------------------------------------------------------

class TestMergeRecommendationTables:

    @pytest.fixture
    def two_tables(self):
        t1 = pa.table({
            "recommended_item": ["Pasta", "Wine"],
            "total_strength":   [3.0, 1.0],
        })
        t2 = pa.table({
            "recommended_item": ["Pasta", "Coffee"],
            "total_strength":   [2.0, 4.0],
        })
        return t1, t2

    def test_single_table_returned_as_is(self, two_tables):
        t1, _ = two_tables
        out = merge_recommendation_tables([t1], n=10)
        assert len(out) == len(t1)

    def test_union_sums_scores(self, two_tables):
        t1, t2 = two_tables
        out = merge_recommendation_tables([t1, t2], strategy="union", n=10)
        pasta_score = next(
            r["total_strength"] for r in out.to_pylist() if r["recommended_item"] == "Pasta"
        )
        assert pasta_score == pytest.approx(5.0)  # 3 + 2

    def test_union_top_item_is_pasta(self, two_tables):
        """Pasta appears in both tables (scores 3+2=5), Coffee only in t2 (4.0).
        Union top item should be Pasta."""
        t1, t2 = two_tables
        out = merge_recommendation_tables([t1, t2], strategy="union", n=10)
        top = out.to_pylist()[0]["recommended_item"]
        assert top == "Pasta"  # 3+2=5 > 4

    def test_best_returns_highest_total_score_table(self, two_tables):
        t1, t2 = two_tables  # t2 total = 6.0, t1 total = 4.0
        out = merge_recommendation_tables([t1, t2], strategy="best", n=10)
        items = [r["recommended_item"] for r in out.to_pylist()]
        assert "Coffee" in items  # Coffee is only in t2

    def test_first_returns_t1(self, two_tables):
        t1, t2 = two_tables
        out = merge_recommendation_tables([t1, t2], strategy="first", n=10)
        items = [r["recommended_item"] for r in out.to_pylist()]
        assert items == ["Pasta", "Wine"]

    def test_n_respected(self, two_tables):
        t1, t2 = two_tables
        out = merge_recommendation_tables([t1, t2], strategy="union", n=1)
        assert len(out) == 1

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            merge_recommendation_tables([])

    def test_invalid_strategy_raises(self, two_tables):
        t1, t2 = two_tables
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            merge_recommendation_tables([t1, t2], strategy="magic")


# ---------------------------------------------------------------------------
# load_interactions with facet_mode="composite"
# ---------------------------------------------------------------------------

class TestLoadInteractionsFacetComposite:

    def test_composite_loads_without_error(self, raw_df):
        with Duwhal() as db:
            count = db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="composite",
            )
        assert count > 0

    def test_composite_set_ids_are_composite_strings(self, raw_df):
        with Duwhal() as db:
            db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="composite",
            )
            set_ids = db.sql("SELECT DISTINCT set_id FROM interactions").column("set_id").to_pylist()
        assert any("|" in sid for sid in set_ids)

    def test_composite_recommendations_work(self, raw_df):
        with Duwhal() as db:
            db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="composite",
            )
            recs = db.recommend(["Pasta"], strategy="graph", n=5)
        assert len(recs) > 0

    def test_invalid_facet_mode_raises(self, raw_df):
        with pytest.raises(ValueError, match="Unknown facet_mode"):
            with Duwhal() as db:
                db.load_interactions(
                    raw_df, set_col="order_id", node_col="item",
                    facet_cols=["region"], facet_mode="bad_mode",
                )


# ---------------------------------------------------------------------------
# load_interactions with facet_mode="entity"
# ---------------------------------------------------------------------------

class TestLoadInteractionsFacetEntity:

    def test_entity_loads_without_error(self, raw_df):
        with Duwhal() as db:
            count = db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="entity",
            )
        assert count > len(raw_df)

    def test_facet_pseudo_nodes_in_db(self, raw_df):
        with Duwhal() as db:
            db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="entity",
            )
            nodes = db.sql("SELECT DISTINCT node_id FROM interactions").column("node_id").to_pylist()
        assert any(n.startswith("facet:") for n in nodes)

    def test_entity_recommendations_from_item_seed(self, raw_df):
        with Duwhal() as db:
            db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="entity",
            )
            recs = db.recommend(["Pasta"], strategy="graph", n=8)
        assert len(recs) > 0

    def test_entity_recommendations_from_facet_seed(self, raw_df):
        with Duwhal() as db:
            db.load_interactions(
                raw_df, set_col="order_id", node_col="item",
                facet_cols=["region"], facet_mode="entity",
            )
            recs = db.recommend(["facet:region=EU"], strategy="graph", n=5)
        items = [r["recommended_item"] for r in recs.to_pylist()
                 if not r["recommended_item"].startswith("facet:")]
        assert len(items) > 0

    def test_no_facets_unchanged_behaviour(self, raw_df):
        with Duwhal() as db:
            count = db.load_interactions(raw_df, set_col="order_id", node_col="item")
        assert count == len(raw_df)


# ---------------------------------------------------------------------------
# recommend_by_facet — basic contract
# ---------------------------------------------------------------------------

class TestRecommendByFacet:

    def test_returns_dict(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        assert isinstance(results, dict)

    def test_global_key_always_present(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        assert "global" in results

    def test_slice_keys_use_col_eq_val_format(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        non_global = [k for k in results if k != "global"]
        assert all("region=" in k for k in non_global)

    def test_facet_keys_present(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        assert "region=EU" in results or "region=BR" in results

    def test_global_result_is_arrow_table(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        assert isinstance(results["global"], pa.Table)

    def test_global_recs_correct(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region", n=10
            )
        items = [r["recommended_item"] for r in results["global"].to_pylist()]
        assert "Croissant" in items or "Granola Bar" in items

    def test_n_parameter_respected(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Pasta"], raw_df, "order_id", "item", "region", n=1
            )
        assert len(results["global"]) <= 1


# ---------------------------------------------------------------------------
# recommend_by_facet — coarsening fallback (the partition problem fix)
# ---------------------------------------------------------------------------

class TestRecommendByFacetFallback:

    def test_empty_slice_receives_fallback_not_empty_table(self, raw_df):
        """Coffee never appears in EU orders.  After coarsening, the EU slice
        should fall back to the global model (non-empty) instead of returning [].
        """
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        eu_items = [r["recommended_item"] for r in results["region=EU"].to_pylist()]
        assert len(eu_items) > 0, "EU slice should get fallback recs, not empty table"

    def test_non_empty_slice_not_affected_by_fallback(self, raw_df):
        """BR slice has Coffee — it should use its own model, not any fallback."""
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region"
            )
        br_items = [r["recommended_item"] for r in results["region=BR"].to_pylist()]
        assert len(br_items) > 0

    def test_multi_facet_fallback_coarsens_to_single_dimension(self, raw_df):
        """With facet_cols=[region, day_period]:
        EU/morning is empty for seed Pasta (EU orders only have night).
        Coarsening should find the EU (any period) slice and use it.
        """
        # Pasta appears in EU/night but NOT in EU/morning
        # At granularity+1: EU (any period) DOES contain Pasta → non-empty
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Pasta"], raw_df, "order_id", "item",
                facet_cols=["region", "day_period"],
            )
        # The EU|morning slice does not exist in raw_df (EU only has night)
        # But we expect no key to return an empty empty-empty table for Pasta-related seeds
        pasta_related = [
            items
            for label, tbl in results.items()
            if label != "global"
            for items in [[r["recommended_item"] for r in tbl.to_pylist()]]
            if len(items) > 0
        ]
        assert len(pasta_related) > 0

    def test_fallback_merge_union_is_default(self, raw_df):
        """Triggering a multi-candidate fallback should not crash with the default merge."""
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region",
                fallback_merge="union",
            )
        assert isinstance(results["region=EU"], pa.Table)

    def test_fallback_merge_best(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region",
                fallback_merge="best",
            )
        assert isinstance(results["region=EU"], pa.Table)

    def test_fallback_merge_first(self, raw_df):
        with Duwhal() as db:
            results = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", "region",
                fallback_merge="first",
            )
        assert isinstance(results["region=EU"], pa.Table)

    def test_list_facet_cols_accepted(self, raw_df):
        """Passing facet_cols as a list should work identically to a string."""
        with Duwhal() as db:
            results_list = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", facet_cols=["region"]
            )
            results_str = db.recommend_by_facet(
                ["Coffee"], raw_df, "order_id", "item", facet_cols="region"
            )
        assert set(results_list.keys()) == set(results_str.keys())


# ---------------------------------------------------------------------------
# validate_disjoint
# ---------------------------------------------------------------------------

class TestValidationDisjoint:

    def test_composite_key_collision_raises(self, raw_df):
        with pytest.raises(ValueError, match="column collision"):
            build_composite_key(raw_df, "region", ["region"])

    def test_facet_entities_collision_raises(self, raw_df):
        with pytest.raises(ValueError, match="column collision"):
            build_facet_entities(raw_df, "item", "item", ["region"])

    def test_split_by_facet_collision_raises(self, raw_df):
        with pytest.raises(ValueError, match="column collision"):
            split_by_facet(raw_df, "order_id", "region", ["region"])

    def test_load_interactions_collision_raises(self, raw_df):
        with Duwhal() as db:
            with pytest.raises(ValueError, match="column collision"):
                db.load_interactions(raw_df, set_col="item", node_col="item")

    def test_extra_cols_collision_raises(self, raw_df):
        with pytest.raises(ValueError, match="column collision"):
            split_by_facet(raw_df, "order_id", "item", "region", extra_cols=["item"])
