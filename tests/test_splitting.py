"""Tests for train/test splitting strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from duwhal.evaluation.splitting import temporal_split, random_split


@pytest.fixture
def timestamped_df():
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(100):
        t = base + timedelta(days=i)
        rows.append({"order_id": f"order_{i}", "item_id": "item_a", "ts": t})
        rows.append({"order_id": f"order_{i}", "item_id": "item_b", "ts": t})
    return pd.DataFrame(rows)


@pytest.fixture
def basic_df():
    rows = [(f"order_{i}", f"item_{i % 5}") for i in range(100)]
    return pd.DataFrame(rows, columns=["order_id", "item_id"])


class TestTemporalSplit:

    def test_returns_two_dataframes(self, timestamped_df):
        train, test = temporal_split(timestamped_df, timestamp_col="ts", set_col="order_id")
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_overlap_between_train_and_test(self, timestamped_df):
        train, test = temporal_split(timestamped_df, timestamp_col="ts", set_col="order_id")
        train_orders = set(train["order_id"])
        test_orders = set(test["order_id"])
        assert train_orders.isdisjoint(test_orders)

    def test_combined_size_equals_original(self, timestamped_df):
        n_orders = timestamped_df["order_id"].nunique()
        train, test = temporal_split(timestamped_df, timestamp_col="ts", set_col="order_id")
        assert train["order_id"].nunique() + test["order_id"].nunique() == n_orders

    def test_test_fraction_approximate(self, timestamped_df):
        n_orders = timestamped_df["order_id"].nunique()
        _, test = temporal_split(timestamped_df, timestamp_col="ts", test_fraction=0.2, set_col="order_id")
        actual_frac = test["order_id"].nunique() / n_orders
        assert abs(actual_frac - 0.2) < 0.05

    def test_train_orders_are_older(self, timestamped_df):
        train, test = temporal_split(timestamped_df, timestamp_col="ts", set_col="order_id")
        max_train_time = train.groupby("order_id")["ts"].min().max()
        min_test_time = test.groupby("order_id")["ts"].min().min()
        assert max_train_time <= min_test_time

    def test_invalid_fraction_raises(self, timestamped_df):
        with pytest.raises(ValueError):
            temporal_split(timestamped_df, "ts", test_fraction=0, set_col="order_id")
        with pytest.raises(ValueError):
            temporal_split(timestamped_df, "ts", test_fraction=1.0, set_col="order_id")


class TestRandomSplit:

    def test_returns_two_dataframes(self, basic_df):
        train, test = random_split(basic_df, set_col="order_id")
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_overlap(self, basic_df):
        train, test = random_split(basic_df, set_col="order_id")
        assert set(train["order_id"]).isdisjoint(set(test["order_id"]))

    def test_combined_equals_original(self, basic_df):
        n = basic_df["order_id"].nunique()
        train, test = random_split(basic_df, set_col="order_id")
        assert train["order_id"].nunique() + test["order_id"].nunique() == n

    def test_test_fraction_approximate(self, basic_df):
        n = basic_df["order_id"].nunique()
        _, test = random_split(basic_df, test_fraction=0.3, set_col="order_id")
        actual = test["order_id"].nunique() / n
        assert abs(actual - 0.3) < 0.1

    def test_seed_is_reproducible(self, basic_df):
        _, test1 = random_split(basic_df, seed=42, set_col="order_id")
        _, test2 = random_split(basic_df, seed=42, set_col="order_id")
        assert set(test1["order_id"]) == set(test2["order_id"])

    def test_different_seeds_differ(self, basic_df):
        _, test1 = random_split(basic_df, seed=1, set_col="order_id")
        _, test2 = random_split(basic_df, seed=999, set_col="order_id")
        # Very unlikely to be identical
        assert set(test1["order_id"]) != set(test2["order_id"])

    def test_invalid_fraction_raises(self, basic_df):
        with pytest.raises(ValueError):
            random_split(basic_df, test_fraction=0, set_col="order_id")
        with pytest.raises(ValueError):
            random_split(basic_df, test_fraction=1.0, set_col="order_id")
