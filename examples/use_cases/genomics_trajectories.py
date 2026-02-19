
"""
Universal Ontology Case: Genomics & Personalized Medicine.
Demonstrates using the InteractionGraph for non-retail data.
In this case: Context = Sample (Patient), Node = Gene/Mutation.
"""

import pandas as pd
from duwhal import InteractionGraph
from duwhal.datasets import generate_genomics_data

def genomics_use_case():
    df = generate_genomics_data()
    
    print("Initializing Genome Interaction Graph...")
    with InteractionGraph() as graph:
        # Step 1: Load using domain-agnostic interface
        graph.load_interactions(df, context_col="sample_id", node_col="gene_id")
        
        # Step 2: Build Topology
        graph.build_topology(min_interactions=2)
        
        # Step 3: Probabilistic Path Traversal
        # We want to find genes most likely to co-occur with 'BRCA1'
        print("\nSearching for Gene Path Trajectories starting at 'BRCA1'...")
        related_genes = graph.rank_nodes(["BRCA1"], steps=2, limit=5)
        
        print("\nRelated Genes Map:")
        print(related_genes)
        
        # The 'reason' metadata shows the most probable molecular pathway / co-occurrence chain
        rows = related_genes.to_pylist()
        if rows:
            best_hit = rows[0]
            print(f"\nStrongest Association: {best_hit['node']}")
            print(f"Molecular Path: {best_hit['reason']}")
        else:
            print("\nNo associations found.")

if __name__ == "__main__":
    genomics_use_case()
