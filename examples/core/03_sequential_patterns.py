
"""
Example of Sequential Pattern Mining.
Detect patterns like "A -> B" ordered by time.
"""

import pandas as pd
from duwhal import Duwhal

# Time-ordered transactions
data = pd.DataFrame({
    "session_id": ["S1", "S1", "S2", "S2", "S3", "S3", "S4"],
    "page_view": [
        "Home", "Checkout",        # S1: Home -> Checkout
        "Home", "Cart",            # S2: Home -> Cart
        "Home", "Checkout",        # S3: Home -> Checkout
        "Search"                   # S4: Search
    ],
    "ts": [
        "2024-01-01 10:00:00", "2024-01-01 10:05:00",
        "2024-01-01 11:00:00", "2024-01-01 11:02:00",
        "2024-01-01 12:00:00", "2024-01-01 12:10:00",
        "2024-01-01 13:00:00"
    ]
})

with Duwhal() as db:
    # 1. Custom Ingestion (since we have timestamps)
    # We load source dataframe first
    import duckdb
    import narwhals as nw
    
    # We can register the dataframe manually to handle the timestamp column
    # Duwhal's load_transactions only keeps order_id and item_id by default
    
    # For sequential patterns, we need the timestamp in the table.
    # The current load_transactions normalizes to (order_id, item_id).
    
    # We can use the lower-level connection to create a custom table
    # or rely on load_transactions if it supported passthrough columns (it currently doesn't).
    
    # Let's do it manually via the connection wrapper for this advanced use case:
    db.conn.register("source_data", nw.from_native(data).to_native())
    db.conn.execute("""
        CREATE OR REPLACE TABLE interactions AS 
        SELECT 
            session_id::VARCHAR AS set_id, 
            page_view::VARCHAR AS node_id,
            ts::TIMESTAMP AS ts
        FROM source_data
    """)
    
    print("Mining Sequential Patterns (A -> B)...")
    # min_support=0.5 means pattern must appear in 50% of sessions
    patterns = db.sequential_patterns(
        timestamp_col="ts",
        min_support=0.4,
        max_gap=None # Any time gap is allowed
    )
    
    print(f"Found {patterns.num_rows} patterns.")
    print(patterns.to_pandas())
    
    # We should see Home -> Checkout with high support
