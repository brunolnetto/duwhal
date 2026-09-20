"""Batch tests for popularity recommendation."""

import pyarrow as pa
import pytest

from duwhal.recommenders.popularity import PopularityRecommender


def _basket_rows(table: pa.Table, basket_id: int) -> list[dict]:
    return [
        {
            key: value
            for key, value in row.items()
            if key != "basket_id"
        }
        for row in table.to_pylist()
        if row["basket_id"] == basket_id
    ]


def test_popularity_batch_matches_scalar(loaded_conn):
    pop = PopularityRecommender(loaded_conn).fit()

    exclusions = [
        None,
        ["bread"],
        ["milk", "bread"],
    ]

    batch = pop.recommend_batch(
        exclusions,
        n=5,
    )

    assert isinstance(batch, pa.Table)
    assert batch.column_names == [
        "basket_id",
        "item_id",
        "score",
        "rank",
    ]

    for basket_id, exclude_items in enumerate(exclusions):
        scalar = pop.recommend(
            n=5,
            exclude_items=exclude_items,
        ).to_pylist()

        assert _basket_rows(batch, basket_id) == scalar


def test_popularity_batch_exclusions_are_per_basket(loaded_conn):
    pop = PopularityRecommender(loaded_conn).fit()

    batch = pop.recommend_batch(
        [
            ["bread"],
            ["milk"],
        ],
        n=10,
    )

    first = {
        row["item_id"]
        for row in _basket_rows(batch, 0)
    }

    second = {
        row["item_id"]
        for row in _basket_rows(batch, 1)
    }

    assert "bread" not in first
    assert "milk" not in second


def test_popularity_batch_preserves_global_rank(loaded_conn):
    pop = PopularityRecommender(loaded_conn).fit()

    scalar = pop.recommend(
        n=10,
        exclude_items=["bread"],
    ).to_pylist()

    batch = pop.recommend_batch(
        [["bread"]],
        n=10,
    )

    assert _basket_rows(batch, 0) == scalar


def test_popularity_batch_empty_input(loaded_conn):
    pop = PopularityRecommender(loaded_conn)

    result = pop.recommend_batch([])

    assert result.num_rows == 0
    assert result.column_names == [
        "basket_id",
        "item_id",
        "score",
        "rank",
    ]


def test_popularity_batch_validates_n(loaded_conn):
    pop = PopularityRecommender(loaded_conn)

    with pytest.raises(ValueError, match="n must be >= 1"):
        pop.recommend_batch([None], n=0)


def test_duwhal_explicit_popularity_batch_api(duwhal_instance):
    db = duwhal_instance
    db.fit_popularity()

    result = db.recommend_popular_batch(
        [
            ["bread"],
            ["milk"],
        ],
        n=5,
    )

    assert isinstance(result, pa.Table)
    assert set(result.column("basket_id").to_pylist()) == {0, 1}


def test_duwhal_generic_popularity_batch_preserves_scalar_semantics(
    duwhal_instance,
):
    db = duwhal_instance
    db.fit_popularity()

    # Popularity does not consume seed items.  The generic API therefore uses
    # seeds_list only to define basket cardinality unless per-basket exclusions
    # are supplied explicitly.
    seeds_list = [
        ["milk"],
        ["bread"],
    ]

    result = db.recommend_batch(
        seeds_list,
        strategy="popular",
        n=5,
    )

    expected = db.recommend_popular(n=5).to_pylist()

    assert _basket_rows(result, 0) == expected
    assert _basket_rows(result, 1) == expected


def test_duwhal_generic_popularity_batch_per_basket_exclusions(
    duwhal_instance,
):
    db = duwhal_instance
    db.fit_popularity()

    result = db.recommend_batch(
        [[], []],
        strategy="popular",
        n=5,
        exclude_items_list=[
            ["bread"],
            ["milk"],
        ],
    )

    assert _basket_rows(result, 0) == db.recommend_popular(
        n=5,
        exclude_items=["bread"],
    ).to_pylist()

    assert _basket_rows(result, 1) == db.recommend_popular(
        n=5,
        exclude_items=["milk"],
    ).to_pylist()


def test_duwhal_generic_popularity_batch_shared_exclusions(
    duwhal_instance,
):
    db = duwhal_instance
    db.fit_popularity()

    result = db.recommend_batch(
        [[], []],
        strategy="popular",
        n=5,
        exclude_items=["bread"],
    )

    expected = db.recommend_popular(
        n=5,
        exclude_items=["bread"],
    ).to_pylist()

    assert _basket_rows(result, 0) == expected
    assert _basket_rows(result, 1) == expected


def test_duwhal_generic_popularity_batch_rejects_conflicting_exclusions(
    duwhal_instance,
):
    db = duwhal_instance
    db.fit_popularity()

    with pytest.raises(
        ValueError,
        match="exclude_items and exclude_items_list",
    ):
        db.recommend_batch(
            [[], []],
            strategy="popular",
            n=5,
            exclude_items=["bread"],
            exclude_items_list=[
                ["milk"],
                ["bread"],
            ],
        )
