
"""
Basic example of using Duwhal to mine association rules.
"""

import pandas as pd
from duwhal import Duwhal

# 1. Create some sample data (milk, diapers, beer pattern)
data = pd.DataFrame({
    "order_id": [
        "T1", "T1", "T1", 
        "T2", "T2", 
        "T3", "T3", "T3", 
        "T4", 
        "T5", "T5"
    ],
    "item_id": [
        "milk", "diapers", "beer",   # T1
        "milk", "diapers",           # T2
        "milk", "diapers", "beer",   # T3
        "milk",                      # T4
        "diapers", "beer"            # T5
    ]
})

print("Source Data:")
print(data)
print("-" * 30)

# 2. Initialize Duwhal
# Use :memory: for transient analysis, or a path for persistent storage
with Duwhal(database=":memory:") as db:
    
    # 3. Load data
    count = db.load_interactions(data, set_col="order_id", node_col="item_id")
    print(f"Loaded {count} transaction rows.\n")
    
    # 4. Mine Association Rules
    # We want rules with at least 20% support (occur in 20% of baskets)
    # and 50% confidence (if A happens, B happens 50% of time)
    print("Mining Association Rules...")
    rules = db.association_rules(
        min_support=0.2, 
        min_confidence=0.5, 
        min_lift=1.0
    )
    
    # rules is a PyArrow Table
    print(f"Found {rules.num_rows} rules.")
    
    # Convert to Pandas for pretty printing
    print(rules.to_pandas().head(5))
    print("-" * 30)
    
    # 5. Make Recommendations using Rules
    # "Customer bought diapers, what else might they buy?"
    print("\nRecommendations for ['diapers']:")
    recs = db.recommend_by_rules(["diapers"], n=3, metric="lift")
    df_recs = recs.to_pandas()
    # Fix unicode arrow for Windows terminals
    if "rule" in df_recs.columns:
        df_recs["rule"] = df_recs["rule"].str.replace("→", "->")
    print(df_recs)
