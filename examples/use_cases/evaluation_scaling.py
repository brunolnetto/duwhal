
"""
Evaluation on a larger synthetic dataset (~500k transactions).
Demonstrates efficient ingestion via Parquet and model training/inference.
"""

import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from duwhal import Duwhal
from duwhal.datasets import generate_large_scale_data

def run_evaluation():
    t0 = time.time()
    df = generate_large_scale_data()
    print(f"Data Generation: {time.time() - t0:.2f}s")
    
    # Save to Parquet to simulate realistic big data ingestion
    parquet_path = "large_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved to {parquet_path}: {Path(parquet_path).stat().st_size / 1024**2:.2f} MB")
    
    # 2. Initialize Duwhal
    # Use disk-based db to handle larger scale without memory pressure
    db_path = "large_scale.duckdb"
    
    # Clean up old db if exists
    if Path(db_path).exists():
        Path(db_path).unlink()
        
    with Duwhal(database=db_path) as db:
        print("\n--- Ingestion ---")
        t0 = time.time()
        # Load from Parquet directly - very fast
        count = db.load_interactions(parquet_path, set_col="order_id", node_col="item_id")
        print(f"Ingested {count:,} rows in {time.time() - t0:.2f}s")
        
        # 3. Model Training & Recommendations
        
        # A. Graph Pre-computation (Accelerates Itemsets & CF)
        print("\n--- Building Graph Adjacency ---", flush=True)
        t0 = time.time()
        # min_cooccurrence=50 filters noise (0.05% of 100k)
        # We need this to efficiently support mining at >50 support
        db.fit_graph(min_cooccurrence=50)
        print(f"Graph built in {time.time() - t0:.2f}s (enables optimized mining)", flush=True)
        
        # B. Association Rules
        print("\n--- Association Rules ---", flush=True)
        t0 = time.time()
        # Lower support for larger/sparser data
        # Note: The presence of the graph above makes this much faster!
        rules = db.association_rules(min_support=0.002, min_confidence=0.1)
        train_time = time.time() - t0
        print(f"Mined {rules.num_rows:,} rules in {train_time:.2f}s")
        
        sample_item = "I0" # Likely popular due to our distribution
        
        if rules.num_rows > 0:
            print("Top 5 Rules by Lift:")
            print(rules.to_pandas().sort_values("lift", ascending=False).head(5))
            
            # Recommend for a sample item
            print(f"\nRecommendations (Rules) for '{sample_item}':")
            t0 = time.time()
            recs = db.recommend_by_rules([sample_item], n=5)
            print(f"Inference Time: {time.time() - t0:.4f}s")
            print(recs.to_pandas())
        
        # B. ItemCF
        print("\n--- Item Collaborative Filtering ---")
        t0 = time.time()
        db.fit_cf(metric="jaccard", min_cooccurrence=10) # Higher threshold for speed/quality on large data
        print(f"Fitted ItemCF model in {time.time() - t0:.2f}s")
        
        print(f"\nRecommendations (ItemCF) for '{sample_item}':")
        t0 = time.time()
        recs_cf = db.recommend_cf([sample_item], n=5)
        print(f"Inference Time: {time.time() - t0:.4f}s")
        print(recs_cf.to_pandas())

    # Cleanup
    if Path(parquet_path).exists():
        Path(parquet_path).unlink()
    if Path(db_path).exists():
        Path(db_path).unlink()
    if Path(db_path + ".wal").exists(): # extensive cleanup
        Path(db_path + ".wal").unlink()

if __name__ == "__main__":
    run_evaluation()
