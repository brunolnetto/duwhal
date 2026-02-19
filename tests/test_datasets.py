
import pandas as pd
import pytest
from duwhal.datasets.genomics import generate_genomics_data
from duwhal.datasets.media import generate_playlist_data
from duwhal.datasets.nlp import generate_nlp_corpus
from duwhal.datasets.retail import generate_retail_transactions, generate_benchmark_patterns
from duwhal.datasets.scaling import generate_large_scale_data
from duwhal.datasets.scc_synthetic import generate_3scc_dataset
from duwhal.datasets.social import generate_filter_bubble_data

def test_generate_genomics_data():
    df = generate_genomics_data(n_patients=10, n_genes=5, n_generic_mutations=20)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["sample_id", "gene_id"]
    # Check for specific known signals if possible, or just size
    # In generate_genomics_data:
    # Strong signal loop: min(100, n_patients) * 2 rows
    # Noise loop: n_generic_mutations rows
    expected_rows = min(100, 10) * 2 + 20
    assert len(df) == expected_rows

def test_generate_playlist_data():
    df = generate_playlist_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["playlist_id", "song_name"]
    # Based on the code:
    # 12 rows (6 pairs) + 2 bridge rows = 14 rows
    assert len(df) == 14

def test_generate_nlp_corpus():
    df = generate_nlp_corpus()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["sentence_id", "token", "pos"]
    # Based on the code:
    # 18 rows (6 sentences * 3 tokens) + 3 bridge rows = 21 rows
    assert len(df) == 21

def test_generate_retail_transactions():
    df = generate_retail_transactions()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["order_id", "item_name", "timestamp"]
    # The list 'transactions' has 17 items
    assert len(df) == 17

def test_generate_benchmark_patterns():
    df = generate_benchmark_patterns()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["order_id", "item_id"]
    # p1: 4 items * 50 = 200
    # p2: 3 items * 50 = 150
    # p3: 200 items
    # Total = 550
    assert len(df) == 550

def test_generate_large_scale_data():
    n_transactions = 100
    density = 5
    df = generate_large_scale_data(n_transactions=n_transactions, n_items=50, density=density)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["order_id", "item_id"]
    assert len(df) == n_transactions * density

def test_generate_3scc_dataset():
    df, metadata = generate_3scc_dataset(
        nodes_per_scc=5,
        n_transient=2,
        baskets_per_scc=10,
        bridge_baskets=5 
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(metadata, dict)
    assert not df.empty
    assert list(df.columns) == ["basket_id", "product_id"]
    
    # Check metadata keys
    expected_keys = [
        "scc_ranges", "transient_range", "n_nodes", 
        "n_baskets", "n_sccs_expected", "test_baskets", "node_labels"
    ]
    for key in expected_keys:
        assert key in metadata

def test_generate_filter_bubble_data():
    df = generate_filter_bubble_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["user_id", "game_title"]
    # community 1: 6 rows * 50 = 300
    # community 2: 6 rows * 50 = 300
    # bridge: 2 rows * 5 = 10
    # Total = 610
    assert len(df) == 610
