import pyarrow as pa
from typing import List, Optional
from duwhal.core.connection import DuckDBConnection

class ItemCF:
    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        metric: str = "jaccard",
        min_cooccurrence: int = 1,
        top_k_similar: int = 50,
    ):
        if metric not in ["jaccard", "cosine", "lift"]:
            raise ValueError(f"Unknown metric: {metric}")
        self.conn = conn
        self.table_name = table_name
        self.metric = metric
        self.min_cooccurrence = min_cooccurrence
        self.top_k_similar = top_k_similar
        self._fitted = False

    def fit(self):
        # Validation for manually set metric
        if self.metric not in ["jaccard", "cosine", "lift"]:
            raise ValueError(f"Unknown metric: {self.metric}")
            
        # Build co-occurrence matrix and similarity scores
        if self.metric == "jaccard":
            score_expr = "p.cooc / (a_cnt.cnt + b_cnt.cnt - p.cooc)"
        elif self.metric == "cosine":
            score_expr = "p.cooc / (sqrt(a_cnt.cnt) * sqrt(b_cnt.cnt))"
        else: # lift
            total_n = self.conn.execute(f"SELECT COUNT(DISTINCT set_id) FROM {self.table_name}").fetchone()[0]
            score_expr = f"(p.cooc * {total_n}) / (a_cnt.cnt * b_cnt.cnt)"

        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _item_counts AS
            SELECT node_id, COUNT(*) AS cnt FROM {self.table_name} GROUP BY 1
        """)

        self.conn.execute(f"""
            CREATE OR REPLACE TABLE _item_similarity AS
            WITH pairs AS (
                SELECT a.node_id AS item_a, b.node_id AS item_b, COUNT(*) AS cooc
                FROM {self.table_name} a
                JOIN {self.table_name} b ON a.set_id = b.set_id AND a.node_id != b.node_id
                GROUP BY 1, 2
                HAVING cooc >= {self.min_cooccurrence}
            )
            SELECT 
                p.item_a, p.item_b,
                {score_expr} AS score
            FROM pairs p
            JOIN _item_counts a_cnt ON p.item_a = a_cnt.node_id
            JOIN _item_counts b_cnt ON p.item_b = b_cnt.node_id
            QUALIFY row_number() OVER (PARTITION BY p.item_a ORDER BY score DESC) <= {self.top_k_similar}
        """)
        self._fitted = True
        return self

    def get_similar_items(self, item_id: str, n: int = 10) -> pa.Table:
        """Return items similar to a single item."""
        if not self._fitted: self.fit()
        return self.conn.query(f"SELECT item_b AS item_id, score FROM _item_similarity WHERE item_a = '{item_id}' ORDER BY score DESC LIMIT {n}")

    def recommend(self, seed_items: List[str], n: int = 10, exclude_seed: bool = True) -> pa.Table:
        if not self._fitted: self.fit()
        if not seed_items: return pa.Table.from_pylist([], schema=pa.schema([("item_id", pa.string()), ("score", pa.float64())]))
        
        self.conn.register("_seeds", pa.Table.from_pylist([{"node_id": s} for s in seed_items]))
        
        exclude_sql = "AND item_b NOT IN (SELECT node_id FROM _seeds)" if exclude_seed else ""
        
        query = f"""
            SELECT item_b AS item_id, SUM(score) AS score
            FROM _item_similarity
            WHERE item_a IN (SELECT node_id FROM _seeds)
            {exclude_sql}
            GROUP BY 1
            ORDER BY score DESC
            LIMIT {n}
        """
        return self.conn.query(query)
