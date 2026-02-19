
"""
NLP Use Case: Token co-occurrence and Semantic Proximity.
Demonstrates how `duwhal` can be used to analyze large corpora of text by representing
Sentences as 'Sets' and Words as 'Nodes'.
"""

import pandas as pd
from duwhal import Duwhal
from duwhal.datasets import generate_nlp_corpus

def nlp_semantic_graph():
    df = generate_nlp_corpus()
    
    print("--- NLP Semantic Token Graph ---")
    with Duwhal() as db:
        db.load_interactions(df, set_col="sentence_id", node_col="token", sort_col="pos")
        
        # 1. Finding Synonyms / Related Concepts (Graph Proximity)
        # Using Graph Traversal to find tokens that appear in similar semantic contexts.
        print("\n[NLP 1] Semantic Proximity for 'Interest Rates':")
        recs = db.recommend(["Interest Rates"], strategy="graph", scoring="probability", n=5)
        for row in recs.to_pylist():
            print(f"- {row['recommended_item']:20} | Proximity: {row['total_strength']:.4f}")

        # 2. Topic Co-occurrence rules
        # If 'AI' is mentioned, which 'Economy' terms are most likely to follow?
        print("\n[NLP 2] Topic Cross-Pollination Rules (Economy -> AI):")
        rules = db.association_rules(min_support=0.01, min_confidence=0.1)
        # Filter rules where antecedent is Economy
        economy_rules = [r for r in rules.to_pylist() if "Economy" in r['antecedents']]
        for rule in economy_rules:
            print(f"When discussing '{rule['antecedents']}', often mentions '{rule['consequents']}' (Lift: {rule['lift']:.2f})")

        # 3. Text Continuity (Sequential Patterns)
        # If the corpus was a stream of tokens, what follows what?
        # (Using sentence_id as sequence context)
        print("\n[NLP 3] Sequential N-gram Patterns:")
        patterns = db.sequential_patterns(min_support=0.05)
        for p in patterns.to_pylist()[:5]:
            print(f"Token '{p['prefix']}' is frequently followed by '{p['suffix']}'")

if __name__ == "__main__":
    nlp_semantic_graph()
