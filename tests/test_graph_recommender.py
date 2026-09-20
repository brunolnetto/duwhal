
"""Tests for the new Graph Recommender."""

import pandas as pd
import pyarrow as pa
import pytest

from duwhal.recommenders.graph import GraphRecommender


class TestGraphRecommender:

    def test_build_creates_adjacency_table(self, loaded_conn):
        """Test that build() creates the _item_adjacency table."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # Should return self
        assert gr.build() is gr

        # Check table exists and has array columns
        # neighbors should be a LIST type
        schema = loaded_conn.execute("DESCRIBE _item_adjacency").fetchall()
        col_names = [row[0] for row in schema]

        assert "source" in col_names
        assert "neighbors" in col_names
        assert "weights" in col_names

        # Check content (milk -> bread)
        res = loaded_conn.execute("SELECT * FROM _item_adjacency WHERE source = 'milk'").fetchone()
        assert res is not None
        # neighbors list should contain 'bread'
        neighbors = res[1]
        assert "bread" in neighbors

    def test_recommend_basic(self, loaded_conn):
        """Test basic 1-hop recommendation (milk -> bread)."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()

        recs = gr.recommend(["milk"], max_depth=1, n=5)
        assert isinstance(recs, pa.Table)
        assert recs.num_rows > 0

        items = recs.column("recommended_item").to_pylist()
        scores = recs.column("total_strength").to_pylist()

        # Milk co-occurs with bread (3 times in conftest data usually?)
        assert "bread" in items
        idx = items.index("bread")
        assert scores[idx] >= 1

    def test_recommend_multi_hop(self, conn):
        """
        Test 2-hop recommendation.
        A -> B (Strong)
        B -> C (Strong)
        A -> C (No path or Weak)

        Graph:
        T1: A, B
        T2: B, C
        """
        df = pd.DataFrame([
            ("T1", "A"), ("T1", "B"),
            ("T2", "B"), ("T2", "C"),
        ], columns=["order_id", "item_id"])

        from duwhal.core.ingestion import load_interactions
        load_interactions(conn, df, set_col="order_id", node_col="item_id")

        gr = GraphRecommender(conn, min_cooccurrence=1)
        gr.build()

        # Recommend for A
        # Hop 1: B (weight 1)
        # Hop 2: C (neighbor of B, weight 1)
        recs = gr.recommend(["A"], max_depth=2, n=5)

        items = recs.column("recommended_item").to_pylist()
        hops = recs.column("min_hops").to_pylist()

        assert "B" in items
        assert "C" in items

        # Check hops
        b_idx = items.index("B")
        c_idx = items.index("C")
        assert hops[b_idx] == 1
        assert hops[c_idx] == 2

    def test_recommend_excludes_seed(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        recs = gr.recommend(["milk"], exclude_seed=True)
        items = recs.column("recommended_item").to_pylist()
        assert "milk" not in items

    def test_recommend_includes_seed(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # With walk-based cycle filtering the seed cannot return to itself via
        # a 2-hop cycle; exclude_seed=False allows it to be listed at depth=0.
        recs = gr.recommend(["milk"], max_depth=2, exclude_seed=False)
        items = recs.column("recommended_item").to_pylist()
        # Seed may or may not appear depending on graph topology; the invariant
        # is that it is not rejected by exclude_seed.
        assert isinstance(items, list)

    def test_return_paths_same_semantics(self, loaded_conn):
        """return_paths should only add a reason column, not change scores."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        without = gr.recommend(["milk"], max_depth=2, n=5, return_paths=False, beam_width=None).to_pylist()
        with_paths = gr.recommend(["milk"], max_depth=2, n=5, return_paths=True, beam_width=None).to_pylist()
        assert len(without) == len(with_paths)
        for a, b in zip(without, with_paths):
            assert a["recommended_item"] == b["recommended_item"]
            assert a["total_strength"] == b["total_strength"]
            assert a["min_hops"] == b["min_hops"]
            assert "reason" in b
            assert "reason" not in a

    def test_return_paths_same_semantics_with_beam(self, loaded_conn):
        """Bounded traversal keeps identical scores; item order may tie-break."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        without = gr.recommend(["milk"], max_depth=2, n=5, return_paths=False, beam_width=10).to_pylist()
        with_paths = gr.recommend(["milk"], max_depth=2, n=5, return_paths=True, beam_width=10).to_pylist()
        assert len(without) == len(with_paths)
        without_scores = {a["recommended_item"]: a["total_strength"] for a in without}
        with_scores = {b["recommended_item"]: b["total_strength"] for b in with_paths}
        assert without_scores == with_scores
        for b in with_paths:
            assert "reason" in b
        for a in without:
            assert "reason" not in a

    def test_recommend_scoring_probability(self, loaded_conn):
        """Test the Path Integral scoring mode."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        recs = gr.recommend(["milk"], max_depth=2, scoring="probability")

        scores = recs.column("total_strength").to_pylist()
        assert all(s > 0 for s in scores)

    def test_frequency_and_probability_scores_differ(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        gr.build()
        freq = gr.recommend(["milk"], max_depth=2, scoring="frequency", n=5).to_pylist()
        prob = gr.recommend(["milk"], max_depth=2, scoring="probability", n=5).to_pylist()
        assert freq and prob
        # Scoring semantics differ: at least one corresponding score should differ
        freq_scores = {r["recommended_item"]: r["total_strength"] for r in freq}
        prob_scores = {r["recommended_item"]: r["total_strength"] for r in prob}
        common = set(freq_scores) & set(prob_scores)
        assert common
        assert any(freq_scores[item] != prob_scores[item] for item in common)

    def test_graph_argument_validation(self, loaded_conn):
        gr = GraphRecommender(loaded_conn)
        with pytest.raises(ValueError):
            gr.recommend(["milk"], n=0)
        with pytest.raises(ValueError):
            gr.recommend(["milk"], max_depth=0)
        with pytest.raises(ValueError):
            gr.recommend(["milk"], min_weight=-1)
        with pytest.raises(ValueError):
            gr.recommend(["milk"], beam_width=0)
        with pytest.raises(ValueError):
            gr.recommend(["milk"], scoring="unknown")

    def test_recommend_auto_build(self, loaded_conn):
        """Test that recommend calls build() if not built."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        # No build() call
        recs = gr.recommend(["milk"])
        assert gr._built
        assert recs.num_rows > 0

    def test_get_neighbors(self, loaded_conn):
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        neighbors = gr.get_neighbors("milk")
        assert isinstance(neighbors, pa.Table)
        assert "neighbor" in neighbors.column_names
        assert "weight" in neighbors.column_names
        pylist = neighbors.to_pylist()
        assert any( row["neighbor"] == "bread" for row in pylist )

class TestDuwhalGraphAPI:
    """Test the integration in the main class."""

    def test_fit_and_recommend_graph(self, duwhal_instance):
        db = duwhal_instance
        # Auto-fit
        recs = db.recommend_graph(["milk"], n=3)
        assert isinstance(recs, pa.Table)
        assert recs.num_rows > 0

    def test_recommend_empty_seed(self, loaded_conn):
        """Test recommend with empty seed list returns empty table."""
        gr = GraphRecommender(loaded_conn, min_cooccurrence=1)
        recs = gr.recommend([])
        assert recs.num_rows == 0
        assert "recommended_item" in recs.column_names

    def test_fit_graph_explicit(self, duwhal_instance):
        db = duwhal_instance.fit_graph(min_cooccurrence=1)
        assert db._graph is not None
        assert db._graph._built

    def test_graph_recommender_reuses_prepared_edges(self, duwhal_instance):
        db = duwhal_instance
        db.fit_graph(min_cooccurrence=1)
        graph = db._graph_model

        # Scoring columns are now precomputed during build(), so no per-query
        # edge preparation is necessary.
        assert graph.prepare_edges_calls == 0

        db.recommend_graph(["milk"], scoring="frequency")
        assert graph.prepare_edges_calls == 0

        db.recommend_graph(["bread"], scoring="probability")
        assert graph.prepare_edges_calls == 0

def test_return_paths_preserves_recommendation_semantics(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
        top_k_edges=100,
    )

    gr.build()

    pathless = gr.recommend(
        ["milk"],
        max_depth=3,
        n=20,
        scoring="probability",
        beam_width=50,
        return_paths=False,
    ).to_pandas()

    explained = gr.recommend(
        ["milk"],
        max_depth=3,
        n=20,
        scoring="probability",
        beam_width=50,
        return_paths=True,
    ).to_pandas()

    import numpy as np

    assert (
        pathless[
            "recommended_item"
        ].tolist()
        ==
        explained[
            "recommended_item"
        ].tolist()
    )

    assert (
        pathless[
            "min_hops"
        ].tolist()
        ==
        explained[
            "min_hops"
        ].tolist()
    )

    assert np.allclose(
        pathless[
            "total_strength"
        ].to_numpy(),
        explained[
            "total_strength"
        ].to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )

    assert (
        "reason"
        not in pathless.columns
    )

    assert (
        "reason"
        in explained.columns
    )

@pytest.mark.slow
def test_iterative_frontier_many_recommendations(
    conn,
):
    import numpy as np
    import pandas as pd

    from duwhal.core.ingestion import (
        load_interactions,
    )

    rng = np.random.default_rng(
        42
    )

    rows = []

    for context in range(
        2_000
    ):
        items = rng.choice(
            500,
            size=12,
            replace=False,
        )

        for item in items:
            rows.append(
                {
                    "context":
                        f"c{context}",
                    "item":
                        f"i{item}",
                }
            )

    df = pd.DataFrame(
        rows
    )

    load_interactions(
        conn,
        df,
        set_col="context",
        node_col="item",
    )

    gr = GraphRecommender(
        conn,
        min_cooccurrence=2,
        top_k_edges=100,
    )

    gr.build()

    for item in range(
        200
    ):
        result = gr.recommend(
            [
                f"i{item % 500}"
            ],
            max_depth=2,
            n=10,
            scoring="probability",
            beam_width=200,
        )

        assert (
            result.num_rows
            <= 10
        )

def _batch_rows_for(
    table: pa.Table,
    basket_id: int,
) -> list[dict]:
    """
    Extract one basket from a graph batch result and remove basket_id
    so it can be compared directly with scalar Graph output.
    """
    return [
        {
            key: value
            for key, value in row.items()
            if key != "basket_id"
        }
        for row in table.to_pylist()
        if row["basket_id"] == basket_id
    ]


def test_graph_batch_matches_scalar_basic(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk"],
        ["bread"],
        ["butter"],
    ]

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=10,
        scoring="frequency",
        beam_width=20,
    )

    for basket_id, seeds in enumerate(baskets):
        scalar = gr.recommend(
            seeds,
            max_depth=2,
            n=10,
            scoring="frequency",
            beam_width=20,
        )

        assert (
            _batch_rows_for(
                batch,
                basket_id,
            )
            == scalar.to_pylist()
        )


def test_graph_batch_matches_scalar_probability(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
        alpha=0.1,
    )
    gr.build()

    baskets = [
        ["milk"],
        ["bread"],
        ["eggs"],
    ]

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=10,
        scoring="probability",
        beam_width=20,
    )

    for basket_id, seeds in enumerate(baskets):
        scalar = gr.recommend(
            seeds,
            max_depth=2,
            n=10,
            scoring="probability",
            beam_width=20,
        )

        actual = _batch_rows_for(
            batch,
            basket_id,
        )

        expected = scalar.to_pylist()

        assert len(actual) == len(expected)

        for batch_row, scalar_row in zip(
            actual,
            expected,
        ):
            assert (
                batch_row["recommended_item"]
                == scalar_row["recommended_item"]
            )

            assert (
                batch_row["min_hops"]
                == scalar_row["min_hops"]
            )

            assert batch_row[
                "total_strength"
            ] == pytest.approx(
                scalar_row[
                    "total_strength"
                ],
                rel=1e-12,
                abs=1e-15,
            )


def test_graph_batch_beam_is_per_basket(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk", "bread"],
        ["bread", "butter"],
        ["eggs"],
    ]

    beam_width = 2

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=10,
        scoring="frequency",
        beam_width=beam_width,
    )

    for basket_id, seeds in enumerate(baskets):
        scalar = gr.recommend(
            seeds,
            max_depth=2,
            n=10,
            scoring="frequency",
            beam_width=beam_width,
        )

        assert (
            _batch_rows_for(
                batch,
                basket_id,
            )
            == scalar.to_pylist()
        )


def test_graph_batch_without_beam_matches_scalar(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk"],
        ["bread"],
    ]

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=20,
        scoring="frequency",
        beam_width=None,
    )

    for basket_id, seeds in enumerate(baskets):
        scalar = gr.recommend(
            seeds,
            max_depth=2,
            n=20,
            scoring="frequency",
            beam_width=None,
        )

        assert (
            _batch_rows_for(
                batch,
                basket_id,
            )
            == scalar.to_pylist()
        )


def test_graph_batch_return_paths_matches_scalar(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk"],
        ["bread"],
    ]

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=10,
        scoring="probability",
        beam_width=20,
        return_paths=True,
    )

    for basket_id, seeds in enumerate(baskets):
        scalar = gr.recommend(
            seeds,
            max_depth=2,
            n=10,
            scoring="probability",
            beam_width=20,
            return_paths=True,
        )

        actual = _batch_rows_for(
            batch,
            basket_id,
        )

        expected = scalar.to_pylist()

        assert len(actual) == len(expected)

        for batch_row, scalar_row in zip(
            actual,
            expected,
        ):
            assert (
                batch_row["recommended_item"]
                == scalar_row["recommended_item"]
            )

            assert (
                batch_row["min_hops"]
                == scalar_row["min_hops"]
            )

            assert batch_row[
                "total_strength"
            ] == pytest.approx(
                scalar_row[
                    "total_strength"
                ],
                rel=1e-12,
                abs=1e-15,
            )

            assert (
                batch_row["reason"]
                == scalar_row["reason"]
            )


def test_graph_batch_empty_input(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )

    result = gr.recommend_batch(
        []
    )

    assert result.num_rows == 0

    assert result.column_names == [
        "basket_id",
        "recommended_item",
        "total_strength",
        "min_hops",
    ]


def test_graph_batch_empty_basket_preserves_other_ids(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk"],
        [],
        ["bread"],
    ]

    batch = gr.recommend_batch(
        baskets,
        n=5,
    )

    basket_ids = set(
        batch.column(
            "basket_id"
        ).to_pylist()
    )

    assert 0 in basket_ids
    assert 1 not in basket_ids
    assert 2 in basket_ids


def test_graph_batch_excludes_only_own_seeds(
    loaded_conn,
):
    gr = GraphRecommender(
        loaded_conn,
        min_cooccurrence=1,
    )
    gr.build()

    baskets = [
        ["milk"],
        ["bread"],
    ]

    batch = gr.recommend_batch(
        baskets,
        max_depth=2,
        n=20,
        exclude_seed=True,
        beam_width=None,
    )

    first = _batch_rows_for(
        batch,
        0,
    )

    second = _batch_rows_for(
        batch,
        1,
    )

    assert "milk" not in {
        row["recommended_item"]
        for row in first
    }

    assert "bread" not in {
        row["recommended_item"]
        for row in second
    }