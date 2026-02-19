
"""
Equilibrium Use Case: Identifying Ecosystem Sinks and Filter Bubbles.
Demonstrates how to use Sink Strongly Connected Components (SCCs) to find 
self-sustaining communities where users/entities get 'trapped'.
"""

import pandas as pd
from duwhal import InteractionGraph
from duwhal.datasets import generate_filter_bubble_data

def equilibrium_analysis():
    df = generate_filter_bubble_data()
    
    print("--- Graph Equilibrium & Community Stability ---")
    with InteractionGraph() as graph:
        graph.load_interactions(df, context_col="user_id", node_col="game_title")
        
        print("\nNode Counts:")
        print(graph.db.sql("SELECT node_id, count(*) FROM interactions GROUP BY 1 ORDER BY 2 DESC").to_pandas())
        # min_confidence=0.1 will catch Generic_Game -> Mario (p=1.0)
        # but drop Mario -> Generic_Game (p=5/105 = 0.047)
        print("\nIdentifying Sink SCCs (Self-Sustaining Filter Bubbles)...")
        equilibrium = graph.find_equilibrium_communities(min_cooccurrence=1, min_confidence=0.1)
        
        if equilibrium.num_rows == 0:
            print("No equilibrium communities found.")
            return

        # Step 2: Display results grouped by SCC
        df_eq = equilibrium.to_pandas()
        for scc_id, group in df_eq.groupby("scc_id"):
            members = group["node"].tolist()
            print(f"\n[Bubble #{scc_id}] Size: {len(members)}")
            print(f"Nodes: {members}")
            
        # Step 3: Probabilistic Traversal from a transient node
        # 'Mario' is part of a sink. 'Generic_Game' links to 'Mario' but 'Mario' rarely links back.
        print("\n[Analysis] Traversal from 'Generic_Game' (Transient) vs 'Mario' (Sink Core):")
        
        print("\nStarting at 'Generic_Game' (Expected to leak into Retro Bubble):")
        res_transient = graph.rank_nodes(["Generic_Game"], steps=2, scoring="probability", limit=5)
        for row in res_transient.to_pylist():
            print(f"- {row['node']:15} | Score: {row['score']:.4f}")
            
        print("\nStarting at 'Mario' (Expected to stay in Retro Sink):")
        res_sink = graph.rank_nodes(["Mario"], steps=2, scoring="probability", limit=5)
        for row in res_sink.to_pylist():
             # Check if the result is in the same bubble
             in_bubble = row['node'] in ["Zelda", "Metroid", "Mario"]
             indicator = "✅ [STABLE]" if in_bubble else "❌ [LEAK]"
             print(f"- {row['node']:15} | Score: {row['score']:.4f} | {indicator}")

if __name__ == "__main__":
    equilibrium_analysis()
