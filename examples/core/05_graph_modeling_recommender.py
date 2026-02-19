"""
Example 05: Graph-Based Recommendations.

Demonstrates how to use the built-in graph recommender for faster lookups
and multi-hop traversals.

The GraphRecommender uses an adjacency list structure (compressed arrays)
instead of massive link tables, reducing storage and speeding up queries.
"""

from duwhal import Duwhal
import pandas as pd
import time

def main():
    # 1. Create Synthetic Data
    # ---------------------------------------------------------
    print("Generating data...")
    transactions = pd.DataFrame([
        # Main Cluster: Milk, Bread, Butter, Eggs
        ("T1", "Milk"), ("T1", "Bread"), ("T1", "Butter"),
        ("T2", "Milk"), ("T2", "Bread"), ("T2", "Eggs"),
        ("T3", "Bread"), ("T3", "Butter"),
        ("T4", "Eggs"), ("T4", "Milk"),
        
        # Secondary Cluster: Beer, Chips, Soda
        ("T5", "Beer"), ("T5", "Chips"),
        ("T6", "Beer"), ("T6", "Soda"),
        ("T7", "Chips"), ("T7", "Soda"),
        
        # Bridge: Someone bought Bread and Beer together (Weak Link)
        ("T8", "Bread"), ("T8", "Beer"),
    ], columns=["order_id", "product_id"])
    
    # 2. Initialize Duwhal
    # ---------------------------------------------------------
    with Duwhal() as db:
        db.load_interactions(transactions, set_col="order_id", node_col="product_id")
        
        # 3. Build Graph
        # ---------------------------------------------------------
        print("\n--- Building Minimum Co-occurrence Graph ---")
        # min_cooccurrence=1 ensures even single shared transactions create an edge
        db.fit_graph(min_cooccurrence=1)
        
        # 4. Standard Recommendation (Direct Neighbors, 2-hop default)
        # ---------------------------------------------------------
        print("\n--- Recommendations for 'Milk' (Default 2-hop) ---")
        recs = db.recommend_graph(["Milk"], n=5)
        print(recs.to_pandas())
        # Expected: Bread, Butter, Eggs (strongest), maybe Beer via Bread (weaker)
        
        # 5. Multi-Hop Discovery
        # ---------------------------------------------------------
        print("\n--- 3-Hop Traversal from 'Milk' ---")
        # Can we reach 'Chips' from 'Milk'?
        # Path: Milk -> Bread -> Beer -> Chips
        recs_deep = db.recommend_graph(["Milk"], max_depth=3, n=10)
        df_deep = recs_deep.to_pandas()
        print(df_deep)
        
        if "Chips" in df_deep["recommended_item"].values:
            print("-> Successfully found 'Chips' via 3-hop connection!")
            
        # 6. Path Integral Scoring (Probabilistic)
        # ---------------------------------------------------------
        print("\n--- Path Integral Scoring (Probability Mode) ---")
        # Scores represent the probability of reaching the item via random walk
        recs_prob = db.recommend_graph(["Milk"], max_depth=3, n=5, scoring="probability")
        print(recs_prob.to_pandas())
            
        # 7. Optimization for Association Rules
        # ---------------------------------------------------------
        print("\n--- Accelerating Association Rules with Graph ---")
        start = time.time()
        # The graph (adjacency list) is already built, so frequent_itemsets
        # will use it to skip the expensive self-join for 2-itemsets.
        rules = db.association_rules(min_support=0.01, min_confidence=0.1)
        print(f"Mined {rules.num_rows} rules in {time.time() - start:.4f}s")
        print(rules.to_pandas().head())

if __name__ == "__main__":
    main()
