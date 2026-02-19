
"""
Benchmarking and Model Comparison.
This example compares different recommendation strategies (Rules, CF, Graph, Popularity)
on the same dataset and evaluates their overlap and explainability.
"""

import pandas as pd
import numpy as np
from duwhal import Duwhal
from duwhal.evaluation import temporal_split, evaluate_recommendations
from duwhal.datasets import generate_benchmark_patterns

def benchmark():
    df = generate_benchmark_patterns()
    print(f"Generated {len(df)} interactions.\n")
    
    with Duwhal() as db:
        db.load_interactions(df, set_col="order_id", node_col="item_id")
        
        # 1. Fit all models
        print("Training models...")
        db.association_rules(min_support=0.01, min_confidence=0.5)
        db.fit_cf(min_cooccurrence=2)
        db.fit_graph(alpha=0.1) # Bayesian smoothing
        db.fit_popularity()
        
        seed = ["Beer"]
        print(f"\nRecommendations for {seed}:")
        
        # Comparison Table
        results = {}
        
        # Association Rules
        recs_rules = db.recommend(seed, strategy="rules", n=3)
        results["Rules"] = recs_rules.column("recommended_item").to_pylist()
        
        # ItemCF
        recs_cf = db.recommend(seed, strategy="cf", n=3)
        results["ItemCF"] = recs_cf.column("recommended_item").to_pylist()
        
        # Graph (Path Integral)
        recs_graph = db.recommend(seed, strategy="graph", scoring="path", n=3)
        results["Graph (Path)"] = recs_graph.column("recommended_item").to_pylist()
        results["Graph Reason"] = recs_graph.column("reason").to_pylist()
        
        # Popularity (Baseline)
        recs_pop = db.recommend(seed, strategy="popular", n=3)
        results["Popularity"] = recs_pop.column("recommended_item").to_pylist()
        
        # Display results
        for model, recs in results.items():
            if model == "Graph Reason": continue
            reason = results.get("Graph Reason")[0] if model == "Graph (Path)" else "N/A"
            print(f"[{model:12}] -> {recs} | Explainability: {reason}")

        # 2. Qualitative Equivalence Research
        print("\n--- Method Equivalence Research ---")
        # In this Beer/Diaper case, Rules and Graph should both strongly suggest Diaper.
        overlap = set(results["Rules"]) & set(results["Graph (Path)"])
        print(f"Overlap between Rules and Graph: {overlap}")
        
        # 3. Path Integral Likelihood
        print("\n--- Basket Likelihood (Path Integral) ---")
        b1 = ["Beer", "Diaper"]
        b2 = ["Beer", "Coke"]
        l1 = db.score_basket(b1)
        l2 = db.score_basket(b2)
        print(f"Likelihood of '{b1}': {l1:.4f} (Strong Cohesion)")
        print(f"Likelihood of '{b2}': {l2:.4f} (Weak Cohesion)")

if __name__ == "__main__":
    benchmark()
