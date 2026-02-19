
"""
Business Use Case: Market Basket Analysis (Retail).
Demonstrates how to use `duwhal` to discover upsell opportunities and bundle packages
using Association Rules and Sequential Patterns.
"""

import pandas as pd
from duwhal import Duwhal
from duwhal.datasets import generate_retail_transactions

def market_basket_analysis():
    df = generate_retail_transactions()
    
    print("--- Retail Business Insights ---")
    with Duwhal() as db:
        db.load_interactions(df, set_col="order_id", node_col="item_name", sort_col="timestamp")
        
        # 1. Discover Product Bundles (Association Rules)
        # We want to find which items 'drag' others into the basket
        print("\n[Strategy 1] Identifying Product Bundles (Association Rules):")
        rules = db.association_rules(min_support=0.01, min_confidence=0.5, min_lift=1.2)
        
        # Sort by Lift to find the strongest correlations (not just popular items)
        for rule in rules.to_pylist()[:5]:
            print(f"Rule: {rule['antecedents']} => {rule['consequents']} (Lift: {rule['lift']:.2f}, Conf: {rule['confidence']:.2f})")

        # 2. Sequential Purchase Behavior
        # Do people buy the Screen Protector AFTER the iPhone?
        print("\n[Strategy 2] Analyzing Sequential Behavior:")
        sequences = db.sequential_patterns(timestamp_col="timestamp", min_support=0.05)
        for seq in sequences.to_pylist()[:5]:
            print(f"Sequence: {seq['prefix']} then {seq['suffix']} (Supp: {seq['support']:.2f})")

        # 3. Contextual Recommendation for a Customer
        # Someone just added Pasta to their cart
        print("\n[Strategy 3] Real-time Cart Recommendations:")
        current_cart = ["Pasta"]
        recs = db.recommend(current_cart, strategy="rules", n=3)
        print(f"Cart: {current_cart} -> Recommendations: {recs.column('recommended_item').to_pylist()}")

if __name__ == "__main__":
    market_basket_analysis()
