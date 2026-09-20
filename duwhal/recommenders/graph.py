from __future__ import annotations

from typing import Any, List, Optional

import pyarrow as pa

from duwhal.core.connection import DuckDBConnection
from duwhal.recommenders._utils import normalize_seeds, seed_table


class GraphRecommender:
    """
    Sparse item-to-item graph recommender.

    Graph construction
    ------------------
    Interactions are projected to distinct ``(set_id, node_id)`` pairs before
    co-occurrence is calculated. Undirected co-occurrences are computed once,
    expanded into directed edges, optionally pruned to ``top_k_edges`` per
    source, and scored during ``build()``.

    Serving
    -------
    Recommendation uses an iterative bounded frontier rather than a recursive
    SQL CTE.

    For each hop:

        frontier
            -> join sparse scored edges
            -> calculate next strengths
            -> ORDER BY strength
            -> LIMIT beam_width
            -> next frontier

    This gives an explicit hard frontier bound and avoids combining recursive
    CTE execution with window functions.

    ``return_paths=False`` keeps the hot-path state minimal:

        (item, strength, depth)

    ``return_paths=True`` additionally carries:

        path

    Path state is explanatory only and does not participate in candidate
    pruning or scoring, so both modes preserve the same recommendation
    semantics.
    """

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        min_cooccurrence: int = 1,
        alpha: float = 0.0,
        top_k_edges: Optional[int] = None,
    ):
        self.conn = conn
        self.table_name = table_name

        self.min_cooccurrence = min_cooccurrence
        self.alpha = alpha
        self.top_k_edges = top_k_edges

        self._built = False
        self._prepared_scoring: str | None = None
        self._prepare_edges_calls = 0
        self._stats = None

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    @property
    def prepare_edges_calls(self) -> int:
        return self._prepare_edges_calls

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self.min_cooccurrence < 1:
            raise ValueError(
                "min_cooccurrence must be >= 1"
            )

        if (
            self.top_k_edges is not None
            and self.top_k_edges < 1
        ):
            raise ValueError(
                "top_k_edges must be >= 1 or None"
            )

    @staticmethod
    def _validate_recommend_params(
        *,
        n: int,
        max_depth: int,
        min_weight: int,
        scoring: str,
        beam_width: Optional[int],
    ) -> None:
        if scoring not in {
            "frequency",
            "probability",
            "path",
        }:
            raise ValueError(
                f"Unknown scoring: {scoring}"
            )

        if n < 1:
            raise ValueError(
                "n must be >= 1"
            )

        if max_depth < 1:
            raise ValueError(
                "max_depth must be >= 1"
            )

        if min_weight < 0:
            raise ValueError(
                "min_weight must be >= 0"
            )

        if (
            beam_width is not None
            and beam_width < 1
        ):
            raise ValueError(
                "beam_width must be >= 1 or None"
            )

    # ------------------------------------------------------------------
    # Graph build
    # ------------------------------------------------------------------

    def build(
        self,
        stats=None,
    ) -> GraphRecommender:
        import time

        self._validate_params()

        start = time.perf_counter()

        # --------------------------------------------------------------
        # Binary/context interaction projection.
        #
        # Raw events remain preserved in the original interactions table.
        # Graph co-occurrence is context-based.
        # --------------------------------------------------------------

        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
                _distinct_interactions AS

            SELECT DISTINCT
                set_id,
                node_id
            FROM {self.table_name}
            """
        )

        # --------------------------------------------------------------
        # Number of contexts containing each item.
        # --------------------------------------------------------------

        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _item_totals AS

            SELECT
                node_id,
                COUNT(*) AS total_interactions
            FROM _distinct_interactions
            GROUP BY node_id
            """
        )

        # --------------------------------------------------------------
        # Compute each undirected pair exactly once.
        # --------------------------------------------------------------

        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
                _item_unordered_pairs AS

            SELECT
                a.node_id AS item_a,
                b.node_id AS item_b,
                COUNT(*) AS cooc

            FROM _distinct_interactions a

            JOIN _distinct_interactions b
              ON a.set_id = b.set_id
             AND a.node_id < b.node_id

            GROUP BY
                a.node_id,
                b.node_id

            HAVING COUNT(*) >= {self.min_cooccurrence}
            """
        )

        # --------------------------------------------------------------
        # Expand once to directed adjacency and optionally retain only
        # top_k_edges for each source.
        #
        # The explicit target tie-break makes the retained adjacency
        # deterministic.
        # --------------------------------------------------------------

        top_k_filter = ""

        if self.top_k_edges is not None:
            top_k_filter = (
                f"WHERE edge_rank <= "
                f"{int(self.top_k_edges)}"
            )

        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE
                _item_adjacency AS

            WITH directed AS (

                SELECT
                    item_a AS source,
                    item_b AS target,
                    cooc
                FROM _item_unordered_pairs

                UNION ALL

                SELECT
                    item_b AS source,
                    item_a AS target,
                    cooc
                FROM _item_unordered_pairs
            ),

            ranked AS (

                SELECT
                    source,
                    target,
                    cooc,

                    ROW_NUMBER() OVER (
                        PARTITION BY source
                        ORDER BY
                            cooc DESC,
                            target
                    ) AS edge_rank

                FROM directed
            )

            SELECT
                source,

                list(
                    target
                    ORDER BY
                        cooc DESC,
                        target
                ) AS neighbors,

                list(
                    cooc
                    ORDER BY
                        cooc DESC,
                        target
                ) AS weights

            FROM ranked

            {top_k_filter}

            GROUP BY source
            """
        )

        # --------------------------------------------------------------
        # Score edges once during model build.
        # --------------------------------------------------------------

        prior_size = (
            self._prior_catalog_size()
        )

        self.conn.execute(
            """
            CREATE OR REPLACE TABLE
                _item_edges_scored AS

            SELECT
                source,
                unnest(neighbors) AS target,
                unnest(weights) AS weight,

                weight::DOUBLE
                    AS frequency_score,

                (
                    weight::DOUBLE + ?
                )
                /
                (
                    t.total_interactions
                    + ? * ?
                )
                    AS probability_score

            FROM _item_adjacency

            JOIN _item_totals t
              ON source = t.node_id
            """,
            [
                self.alpha,
                self.alpha,
                prior_size,
            ],
        )

        self._built = True
        self._prepared_scoring = "frequency"

        duration_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # --------------------------------------------------------------
        # Stats
        # --------------------------------------------------------------

        if stats is not None:
            stats.duration_ms = duration_ms

            nodes = self.conn.execute(
                """
                SELECT COUNT(*)
                FROM _item_adjacency
                """
            ).fetchone()[0]

            edges = (
                self.conn.execute(
                    """
                    SELECT
                        COALESCE(
                            SUM(len(neighbors)),
                            0
                        )
                    FROM _item_adjacency
                    """
                ).fetchone()[0]
                or 0
            )

            degrees = self.conn.execute(
                """
                SELECT
                    COALESCE(
                        AVG(len(neighbors)),
                        0
                    ) AS mean_degree,

                    COALESCE(
                        MAX(len(neighbors)),
                        0
                    ) AS max_degree,

                    COALESCE(
                        PERCENTILE_CONT(0.95)
                        WITHIN GROUP (
                            ORDER BY len(neighbors)
                        ),
                        0
                    ) AS p95_degree

                FROM _item_adjacency
                """
            ).fetchone()

            stats.table_stats = {
                "nodes":
                    nodes,
                "edges":
                    edges,
                "mean_degree":
                    degrees[0] or 0,
                "p95_degree":
                    degrees[2] or 0,
                "max_degree":
                    degrees[1] or 0,
            }

            self._stats = stats

        return self

    def _prior_catalog_size(self) -> int:
        total = self.conn.execute(
            f"""
            SELECT COUNT(DISTINCT node_id)
            FROM {self.table_name}
            """
        ).fetchone()[0]

        return max(
            total,
            1,
        )

    # ------------------------------------------------------------------
    # Edge inspection
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        item_id: str,
    ) -> pa.Table:
        if not self._built:
            self.build()

        self.conn.register(
            "_item_lookup",
            pa.Table.from_pylist(
                [
                    {
                        "node_id":
                            str(item_id),
                    }
                ]
            ),
        )

        return self.conn.query(
            """
            SELECT
                target AS neighbor,
                weight

            FROM _item_edges_scored

            WHERE source = (
                SELECT node_id
                FROM _item_lookup
            )

            ORDER BY
                weight DESC,
                target
            """
        )

    # ------------------------------------------------------------------
    # Serving helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_column(
        scoring: str,
    ) -> str:
        if scoring in {
            "probability",
            "path",
        }:
            return "probability_score"

        return "frequency_score"

    @staticmethod
    def _empty_result(
        *,
        return_paths: bool,
    ) -> pa.Table:
        schema = [
            (
                "recommended_item",
                pa.string(),
            ),
            (
                "total_strength",
                pa.float64(),
            ),
            (
                "min_hops",
                pa.int32(),
            ),
        ]

        if return_paths:
            schema.append(
                (
                    "reason",
                    pa.string(),
                )
            )

        return pa.Table.from_batches(
            [],
            schema=pa.schema(
                schema
            ),
        )


    @staticmethod
    def _empty_batch_result(
        *,
        return_paths: bool,
    ) -> pa.Table:
        """
        Return an empty Graph batch result with a stable schema.
        """
        schema = [
            (
                "basket_id",
                pa.int64(),
            ),
            (
                "recommended_item",
                pa.string(),
            ),
            (
                "total_strength",
                pa.float64(),
            ),
            (
                "min_hops",
                pa.int32(),
            ),
        ]

        if return_paths:
            schema.append(
                (
                    "reason",
                    pa.string(),
                )
            )

        return pa.Table.from_batches(
            [],
            schema=pa.schema(
                schema
            ),
        )

    def _register_batch_seeds(
        self,
        seeds_list: list[Any],
    ) -> int:
        """
        Register all seed baskets in one Arrow relation.

        ``basket_id`` preserves the input order. Seed normalization follows
        the scalar serving path exactly: values are converted to unique,
        non-empty VARCHAR identifiers per basket.
        """
        rows: list[dict[str, Any]] = []

        for basket_id, seed_items in enumerate(
            seeds_list
        ):
            seeds = normalize_seeds(
                seed_items
            )

            for node_id in seeds:
                rows.append(
                    {
                        "basket_id":
                            basket_id,
                        "node_id":
                            node_id,
                    }
                )

        table = pa.Table.from_pylist(
            rows,
            schema=pa.schema(
                [
                    (
                        "basket_id",
                        pa.int64(),
                    ),
                    (
                        "node_id",
                        pa.string(),
                    ),
                ]
            ),
        )

        self.conn.register(
            "_graph_batch_seeds",
            table,
        )

        return len(rows)

    def _initialize_batch_frontier(
        self,
        *,
        return_paths: bool,
    ) -> int:
        """
        Initialize one independent frontier per basket.

        Seed strength is normalized over valid seeds within each basket.
        The counts are calculated with ``GROUP BY`` rather than a serving-path
        window function.
        """
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_batch_valid_seeds AS

            SELECT DISTINCT
                s.basket_id,
                s.node_id

            FROM _graph_batch_seeds s

            JOIN _item_totals i
              ON i.node_id = s.node_id
            """
        )

        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_batch_seed_counts AS

            SELECT
                basket_id,
                COUNT(*) AS seed_count

            FROM _graph_batch_valid_seeds

            GROUP BY basket_id
            """
        )

        if return_paths:
            self.conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE
                    _graph_batch_frontier AS

                SELECT
                    s.basket_id,
                    s.node_id AS item,

                    (
                        1.0::DOUBLE
                        / c.seed_count
                    ) AS strength,

                    0::INTEGER AS depth,
                    [s.node_id] AS path

                FROM _graph_batch_valid_seeds s

                JOIN _graph_batch_seed_counts c
                  USING (basket_id)
                """
            )

        else:
            self.conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE
                    _graph_batch_frontier AS

                SELECT
                    s.basket_id,
                    s.node_id AS item,

                    (
                        1.0::DOUBLE
                        / c.seed_count
                    ) AS strength,

                    0::INTEGER AS depth

                FROM _graph_batch_valid_seeds s

                JOIN _graph_batch_seed_counts c
                  USING (basket_id)
                """
            )

        return self.conn.execute(
            """
            SELECT COUNT(*)
            FROM _graph_batch_frontier
            """
        ).fetchone()[0]

    def _initialize_batch_walks(
        self,
    ) -> None:
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_batch_walks AS

            SELECT *
            FROM _graph_batch_frontier
            """
        )

    def _expand_batch_frontier(
        self,
        *,
        score_col: str,
        min_weight: int,
        beam_width: Optional[int],
        return_paths: bool,
    ) -> int:
        """
        Expand exactly one hop for every basket.

        Beam pruning remains independent for each basket. Instead of a
        ``ROW_NUMBER() OVER (PARTITION BY basket_id ...)`` window, candidates
        are packed into a deterministically ordered LIST per basket, sliced to
        ``beam_width``, then unnested.
        """
        if return_paths:
            candidate_sql = f"""
                SELECT
                    f.basket_id,
                    e.target AS item,

                    (
                        f.strength
                        * e.{score_col}
                    ) AS strength,

                    (
                        f.depth + 1
                    )::INTEGER AS depth,

                    list_append(
                        f.path,
                        e.target
                    ) AS path,

                    f.item AS parent

                FROM _graph_batch_frontier f

                JOIN _item_edges_scored e
                  ON f.item = e.source

                WHERE
                    e.weight >= {int(min_weight)}
            """

        else:
            candidate_sql = f"""
                SELECT
                    f.basket_id,
                    e.target AS item,

                    (
                        f.strength
                        * e.{score_col}
                    ) AS strength,

                    (
                        f.depth + 1
                    )::INTEGER AS depth,

                    f.item AS parent

                FROM _graph_batch_frontier f

                JOIN _item_edges_scored e
                  ON f.item = e.source

                WHERE
                    e.weight >= {int(min_weight)}
            """

        if beam_width is None:
            if return_paths:
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE
                        _graph_batch_next AS

                    SELECT
                        basket_id,
                        item,
                        strength,
                        depth,
                        path

                    FROM (
                        {candidate_sql}
                    ) candidates

                    ORDER BY
                        basket_id,
                        strength DESC,
                        item,
                        parent
                    """
                )

            else:
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE
                        _graph_batch_next AS

                    SELECT
                        basket_id,
                        item,
                        strength,
                        depth

                    FROM (
                        {candidate_sql}
                    ) candidates

                    ORDER BY
                        basket_id,
                        strength DESC,
                        item,
                        parent
                    """
                )

        elif return_paths:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE
                    _graph_batch_next AS

                WITH candidates AS (

                    {candidate_sql}

                ),

                packed AS (

                    SELECT
                        basket_id,

                        list_slice(

                            list(

                                struct_pack(
                                    item := item,
                                    strength := strength,
                                    depth := depth,
                                    path := path,
                                    parent := parent
                                )

                                ORDER BY
                                    strength DESC,
                                    item,
                                    parent

                            ),

                            1,
                            {int(beam_width)}

                        ) AS entries

                    FROM candidates

                    GROUP BY basket_id
                ),

                expanded AS (

                    SELECT
                        basket_id,
                        unnest(entries) AS entry

                    FROM packed
                )

                SELECT
                    basket_id,
                    entry.item AS item,
                    entry.strength AS strength,
                    entry.depth::INTEGER AS depth,
                    entry.path AS path

                FROM expanded
                """
            )

        else:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE
                    _graph_batch_next AS

                WITH candidates AS (

                    {candidate_sql}

                ),

                packed AS (

                    SELECT
                        basket_id,

                        list_slice(

                            list(

                                struct_pack(
                                    item := item,
                                    strength := strength,
                                    depth := depth,
                                    parent := parent
                                )

                                ORDER BY
                                    strength DESC,
                                    item,
                                    parent

                            ),

                            1,
                            {int(beam_width)}

                        ) AS entries

                    FROM candidates

                    GROUP BY basket_id
                ),

                expanded AS (

                    SELECT
                        basket_id,
                        unnest(entries) AS entry

                    FROM packed
                )

                SELECT
                    basket_id,
                    entry.item AS item,
                    entry.strength AS strength,
                    entry.depth::INTEGER AS depth

                FROM expanded
                """
            )

        return self.conn.execute(
            """
            SELECT COUNT(*)
            FROM _graph_batch_next
            """
        ).fetchone()[0]

    def _advance_batch_frontier(
        self,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO _graph_batch_walks

            SELECT *
            FROM _graph_batch_next
            """
        )

        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_batch_frontier AS

            SELECT *
            FROM _graph_batch_next
            """
        )

    def _aggregate_batch_walks(
        self,
        *,
        scoring: str,
        n: int,
        exclude_seed: bool,
        return_paths: bool,
    ) -> pa.Table:
        """
        Aggregate walks and return top-N recommendations independently
        for every basket.
        """
        aggregate = (
            "MAX"
            if scoring == "path"
            else "SUM"
        )

        exclude_sql = ""

        if exclude_seed:
            exclude_sql = """
                AND NOT EXISTS (

                    SELECT 1

                    FROM _graph_batch_valid_seeds s

                    WHERE
                        s.basket_id = w.basket_id
                        AND s.node_id = w.item
                )
            """

        if return_paths:
            scored_sql = f"""
                SELECT
                    w.basket_id,
                    w.item,

                    {aggregate}(
                        w.strength
                    ) AS total_strength,

                    MIN(
                        w.depth
                    )::INTEGER AS min_hops,

                    arg_max(
                        array_to_string(
                            w.path,
                            ' -> '
                        ),
                        w.strength
                    ) AS reason

                FROM _graph_batch_walks w

                WHERE
                    w.depth > 0

                    {exclude_sql}

                GROUP BY
                    w.basket_id,
                    w.item
            """

            packed_entry = """
                struct_pack(
                    recommended_item := item,
                    total_strength := total_strength,
                    min_hops := min_hops,
                    reason := reason
                )
            """

            final_columns = """
                entry.recommended_item
                    AS recommended_item,

                entry.total_strength
                    AS total_strength,

                entry.min_hops::INTEGER
                    AS min_hops,

                entry.reason
                    AS reason
            """

        else:
            scored_sql = f"""
                SELECT
                    w.basket_id,
                    w.item,

                    {aggregate}(
                        w.strength
                    ) AS total_strength,

                    MIN(
                        w.depth
                    )::INTEGER AS min_hops

                FROM _graph_batch_walks w

                WHERE
                    w.depth > 0

                    {exclude_sql}

                GROUP BY
                    w.basket_id,
                    w.item
            """

            packed_entry = """
                struct_pack(
                    recommended_item := item,
                    total_strength := total_strength,
                    min_hops := min_hops
                )
            """

            final_columns = """
                entry.recommended_item
                    AS recommended_item,

                entry.total_strength
                    AS total_strength,

                entry.min_hops::INTEGER
                    AS min_hops
            """

        query = f"""
            WITH scored AS (

                {scored_sql}

            ),

            packed AS (

                SELECT
                    basket_id,

                    list_slice(

                        list(

                            {packed_entry}

                            ORDER BY
                                total_strength DESC,
                                item

                        ),

                        1,
                        {int(n)}

                    ) AS entries

                FROM scored

                GROUP BY basket_id
            ),

            expanded AS (

                SELECT
                    basket_id,
                    unnest(entries) AS entry

                FROM packed
            )

            SELECT
                basket_id,

                {final_columns}

            FROM expanded

            ORDER BY
                basket_id,
                total_strength DESC,
                recommended_item
        """

        return self.conn.query(
            query
        )

    def _initialize_frontier(
        self,
        *,
        return_paths: bool,
    ) -> int:
        """
        Initialize one root per valid seed without window functions.

        Seed strength is normalized over valid seeds only. The valid seed count
        is calculated as a scalar first, avoiding COUNT(*) OVER () in the
        serving path.
        """

        valid_seed_count = self.conn.execute(
            """
            SELECT COUNT(*)

            FROM _seeds s

            JOIN _item_totals i
            ON i.node_id = s.node_id
            """
        ).fetchone()[0]

        if valid_seed_count == 0:
            return 0

        initial_strength = (
            1.0
            / valid_seed_count
        )

        if return_paths:
            self.conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE
                    _graph_frontier AS

                SELECT
                    s.node_id AS item,
                    ?::DOUBLE AS strength,
                    0::INTEGER AS depth,
                    [s.node_id] AS path

                FROM _seeds s

                JOIN _item_totals i
                ON i.node_id = s.node_id
                """,
                [initial_strength],
            )

        else:
            self.conn.execute(
                """
                CREATE OR REPLACE TEMP TABLE
                    _graph_frontier AS

                SELECT
                    s.node_id AS item,
                    ?::DOUBLE AS strength,
                    0::INTEGER AS depth

                FROM _seeds s

                JOIN _item_totals i
                ON i.node_id = s.node_id
                """,
                [initial_strength],
            )

        return valid_seed_count

    def _initialize_walks(
        self,
    ) -> None:
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_walks AS

            SELECT *
            FROM _graph_frontier
            """
        )

    def _expand_frontier(
        self,
        *,
        score_col: str,
        min_weight: int,
        beam_width: Optional[int],
        return_paths: bool,
    ) -> int:
        """
        Expand exactly one graph hop.

        Because _graph_frontier contains one depth only, beam pruning is a
        simple global ORDER BY ... LIMIT. No recursive CTE and no window
        function are required.
        """

        limit_sql = ""

        if beam_width is not None:
            limit_sql = (
                f"LIMIT {int(beam_width)}"
            )

        if return_paths:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE
                    _graph_next AS

                SELECT
                    item,
                    strength,
                    depth,
                    path

                FROM (

                    SELECT
                        e.target AS item,

                        (
                            f.strength
                            * e.{score_col}
                        ) AS strength,

                        (
                            f.depth + 1
                        )::INTEGER AS depth,

                        list_append(
                            f.path,
                            e.target
                        ) AS path,

                        f.item AS parent

                    FROM _graph_frontier f

                    JOIN _item_edges_scored e
                      ON f.item = e.source

                    WHERE
                        e.weight >= {int(min_weight)}
                ) candidates

                ORDER BY
                    strength DESC,
                    item,
                    parent

                {limit_sql}
                """
            )

        else:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE
                    _graph_next AS

                SELECT
                    item,
                    strength,
                    depth

                FROM (

                    SELECT
                        e.target AS item,

                        (
                            f.strength
                            * e.{score_col}
                        ) AS strength,

                        (
                            f.depth + 1
                        )::INTEGER AS depth,

                        f.item AS parent

                    FROM _graph_frontier f

                    JOIN _item_edges_scored e
                      ON f.item = e.source

                    WHERE
                        e.weight >= {int(min_weight)}
                ) candidates

                ORDER BY
                    strength DESC,
                    item,
                    parent

                {limit_sql}
                """
            )

        return self.conn.execute(
            """
            SELECT COUNT(*)
            FROM _graph_next
            """
        ).fetchone()[0]

    def _advance_frontier(
        self,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO _graph_walks
            SELECT *
            FROM _graph_next
            """
        )

        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE
                _graph_frontier AS

            SELECT *
            FROM _graph_next
            """
        )

    def _aggregate_walks(
        self,
        *,
        scoring: str,
        n: int,
        exclude_seed: bool,
        return_paths: bool,
    ) -> pa.Table:
        aggregate = (
            "MAX"
            if scoring == "path"
            else "SUM"
        )

        exclude_sql = ""

        if exclude_seed:
            exclude_sql = """
                AND item NOT IN (
                    SELECT node_id
                    FROM _seeds
                )
            """

        reason_sql = ""

        if return_paths:
            reason_sql = """
                ,
                arg_max(
                    array_to_string(
                        path,
                        ' -> '
                    ),
                    strength
                ) AS reason
            """

        query = f"""
            SELECT
                item AS recommended_item,

                {aggregate}(
                    strength
                ) AS total_strength,

                MIN(
                    depth
                )::INTEGER AS min_hops

                {reason_sql}

            FROM _graph_walks

            WHERE
                depth > 0

                {exclude_sql}

            GROUP BY item

            ORDER BY
                total_strength DESC,
                recommended_item

            LIMIT {int(n)}
        """

        return self.conn.query(
            query
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        seed_items: Any,
        max_depth: int = 2,
        min_weight: int = 1,
        n: int = 10,
        exclude_seed: bool = True,
        scoring: str = "frequency",
        return_paths: bool = False,
        beam_width: Optional[int] = 200,
    ) -> pa.Table:
        if not self._built:
            self.build()

        self._validate_params()

        self._validate_recommend_params(
            n=n,
            max_depth=max_depth,
            min_weight=min_weight,
            scoring=scoring,
            beam_width=beam_width,
        )

        seeds = normalize_seeds(
            seed_items
        )

        if not seeds:
            return self._empty_result(
                return_paths=return_paths
            )

        # Explicit VARCHAR seed schema lives in seed_table().
        self.conn.register(
            "_seeds",
            seed_table(
                seeds
            ),
        )

        # --------------------------------------------------------------
        # Initial frontier
        # --------------------------------------------------------------

        valid_seed_count = (
            self._initialize_frontier(
                return_paths=return_paths
            )
        )

        if valid_seed_count == 0:
            return self._empty_result(
                return_paths=return_paths
            )

        self._initialize_walks()

        score_col = (
            self._score_column(
                scoring
            )
        )

        # --------------------------------------------------------------
        # Iterative bounded walk.
        #
        # Each loop executes one normal SQL query and produces one depth.
        # --------------------------------------------------------------

        for _depth in range(
            1,
            max_depth + 1,
        ):
            next_count = (
                self._expand_frontier(
                    score_col=score_col,
                    min_weight=min_weight,
                    beam_width=beam_width,
                    return_paths=return_paths,
                )
            )

            if next_count == 0:
                break

            self._advance_frontier()

        return self._aggregate_walks(
            scoring=scoring,
            n=n,
            exclude_seed=exclude_seed,
            return_paths=return_paths,
        )


    def recommend_batch(
        self,
        seeds_list: list[Any],
        max_depth: int = 2,
        min_weight: int = 1,
        n: int = 10,
        exclude_seed: bool = True,
        scoring: str = "frequency",
        return_paths: bool = False,
        beam_width: Optional[int] = 200,
    ) -> pa.Table:
        """
        Vectorized graph recommendation for multiple seed baskets.

        All baskets traverse the same scored graph in one iterative serving
        pipeline. ``basket_id`` in the returned table maps each recommendation
        back to the input basket position.
        """
        if not self._built:
            self.build()

        self._validate_params()

        self._validate_recommend_params(
            n=n,
            max_depth=max_depth,
            min_weight=min_weight,
            scoring=scoring,
            beam_width=beam_width,
        )

        if not seeds_list:
            return self._empty_batch_result(
                return_paths=return_paths
            )

        seed_count = (
            self._register_batch_seeds(
                seeds_list
            )
        )

        if seed_count == 0:
            return self._empty_batch_result(
                return_paths=return_paths
            )

        valid_seed_count = (
            self._initialize_batch_frontier(
                return_paths=return_paths
            )
        )

        if valid_seed_count == 0:
            return self._empty_batch_result(
                return_paths=return_paths
            )

        self._initialize_batch_walks()

        score_col = (
            self._score_column(
                scoring
            )
        )

        for _depth in range(
            1,
            max_depth + 1,
        ):
            next_count = (
                self._expand_batch_frontier(
                    score_col=score_col,
                    min_weight=min_weight,
                    beam_width=beam_width,
                    return_paths=return_paths,
                )
            )

            if next_count == 0:
                break

            self._advance_batch_frontier()

        return self._aggregate_batch_walks(
            scoring=scoring,
            n=n,
            exclude_seed=exclude_seed,
            return_paths=return_paths,
        )

    # ------------------------------------------------------------------
    # Basket score
    # ------------------------------------------------------------------

    def score_basket(
        self,
        items: List[str],
    ) -> float:
        if not items:
            return 1.0

        if not self._built:
            self.build()

        recs = self.recommend(
            items[:1],
            max_depth=len(items) + 1,
            exclude_seed=False,
            scoring="probability",
            n=1000,
        )

        others = set(
            items[1:]
        )

        return sum(
            row["total_strength"]
            for row
            in recs.to_pylist()
            if row[
                "recommended_item"
            ]
            in others
        )

