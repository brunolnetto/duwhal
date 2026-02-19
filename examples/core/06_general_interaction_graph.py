"""
Example 06: General-Purpose Interaction Graph (Playlist Generation)

Demonstrates how to use the domain-agnostic `InteractionGraph` class
to model Music Playlists (Context) and Songs (Nodes).
"""

import pandas as pd
from duwhal import InteractionGraph

def main():
    # 1. Create Data: Playlists containing Songs
    print("Generating Music Data...")
    playlists = pd.DataFrame([
        # Rock Playlist
        ("PL_1", "Bohemian Rhapsody"), ("PL_1", "Stairway to Heaven"), ("PL_1", "Hotel California"),
        # Pop Playlist
        ("PL_2", "Thriller"), ("PL_2", "Billie Jean"), ("PL_2", "Hotel California"), # Bridge song!
        # Metal Playlist
        ("PL_3", "Master of Puppets"), ("PL_3", "Enter Sandman"),
        # Discovery
        ("PL_4", "Enter Sandman"), ("PL_4", "Bohemian Rhapsody"), # Bridge between Metal and Rock
    ], columns=["playlist_id", "song_name"])
    
    print(playlists)
    
    # 2. Use the General API (No "Order" or "Product" jargon!)
    # ---------------------------------------------------------
    print("\n--- Modeling Music Graph ---")
    
    # Initialize generic graph wrapper
    graph = InteractionGraph(database=":memory:")
    
    with graph:
        # Load interactions (Context=Playlist, Node=Song)
        graph.load_interactions(
            data=playlists, 
            context_col="playlist_id", 
            node_col="song_name"
        )
        
        # Build topology (Path Integral framework enabled by default)
        graph.build_topology(min_interactions=1)
        
        # 3. Discover Related Songs (Path Integral Scoring)
        # -------------------------------------------------
        seed_song = "Master of Puppets"
        print(f"\nSongs related to '{seed_song}' (via shared playlists):")
        
        # Find 3-hop connections (Metal -> Rock via bridges)
        # using probabilistic scoring
        related = graph.rank_nodes(
            seed_nodes=[seed_song],
            steps=3,
            scoring="probability",
            limit=5
        )
        
        print(related)
        
        if "Hotel California" in related["node"].values:
            print("\n-> Discovery: Found 'Hotel California' (Rock/Pop) from Metal start!")
            print("   Path: Master of Puppets -> Enter Sandman -> Bohemian Rhapsody -> Hotel California")

if __name__ == "__main__":
    main()
