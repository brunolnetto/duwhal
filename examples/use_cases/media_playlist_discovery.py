
"""
Media Use Case: Playlist discovery and Song ranking.
Demonstrates the `InteractionGraph` universal ontology applied to Audio/Media datasets.
"""

import pandas as pd
from duwhal import InteractionGraph
from duwhal.datasets import generate_playlist_data

def media_discovery_use_case():
    df = generate_playlist_data()
    
    print("--- Media & Entertainment Discovery Graph ---")
    with InteractionGraph() as graph:
        # Load using the General Purpose InteractionGraph
        graph.load_interactions(df, context_col="playlist_id", node_col="song_name")
        
        # Build the topology based on co-occurrence in playlists
        graph.build_topology(min_interactions=1)
        
        # Scenario 1: Multi-hop discovery (Expanding a fan's taste)
        # If I like 'Stairway to Heaven' (Rock), what else is in the proximity?
        seed = ["Stairway to Heaven"]
        print(f"\n[Case 1] 2-Hop Discovery for '{seed[0]}':")
        results = graph.rank_nodes(seed, steps=2, scoring="probability", limit=5)
        
        for row in results.to_pylist():
            print(f"- {row['node']:20} | Score: {row['score']:.4f} | Path: {row['reason']}")

        # Scenario 2: Finding a 'Bridge' across genres
        # Using 3-hop traversal to cross the 'Playlist_Bridge'
        seed_jazz = ["Blue in Green"]
        print(f"\n[Case 2] Crossing Genres from Jazz ('{seed_jazz[0]}') using 3-hop traversal:")
        results_jazz = graph.rank_nodes(seed_jazz, steps=3, scoring="probability", limit=5)
        
        for row in results_jazz.to_pylist():
            # We filter for rock songs to see the bridge in action
            is_rock = "Wish" in row['node'] or "Stairway" in row['node'] or "Lotta" in row['node']
            indicator = "⭐ [BRIDGE]" if is_rock else "   "
            print(f"{indicator} {row['node']:20} | Score: {row['score']:.4f}")

if __name__ == "__main__":
    media_discovery_use_case()
