
"""
Example of Collaborative Filtering (Item-to-Item CF) with Duwhal.
"""

import pandas as pd
from duwhal import Duwhal

# Sample purchase history
transactions = pd.DataFrame([
    ("User1", "Laptop"), ("User1", "Mouse"), ("User1", "HDMI Cable"),
    ("User2", "Laptop"), ("User2", "Mouse"),
    ("User3", "Mouse"), ("User3", "HDMI Cable"),
    ("User4", "Laptop"),
    ("User5", "Headphones"), ("User5", "Mouse")
], columns=["order_id", "product"])

with Duwhal() as db:
    db.load_interactions(transactions, set_col="order_id", node_col="product")
    
    # 1. Fit the Item-CF model
    # metric="jaccard" measures overlap between sets of customers who bought items
    # min_cooccurrence=1 means items must appear together at least once
    print("Fitting ItemCF model...")
    db.fit_cf(metric="jaccard", min_cooccurrence=1)
    
    # 2. Get recommendations
    # "Customer is looking at a Laptop, what else is similar?"
    seed_item = "Laptop"
    print(f"\nRecommendations for someone buying '{seed_item}':")
    
    recs = db.recommend_cf([seed_item], n=3)
    
    # Result -> Mouse (bought together by User1, User2)
    print(recs.to_pandas())
    
    # 3. Get recommendations for a basket
    # User has Mouse and HDMI Cable in cart
    basket = ["Mouse", "HDMI Cable"]
    print(f"\nRecommendations for basket {basket}:")
    
    recs_basket = db.recommend_cf(basket, n=3)
    print(recs_basket.to_pandas())
