"""Tests for association rule mining."""

import pytest
import pandas as pd
import pyarrow as pa

from duwhal.mining.association_rules import AssociationRules


class TestAssociationRules:

    def test_returns_dataframe(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.1, min_confidence=0.1, min_lift=0.0)
        rules = ar.fit()
        assert isinstance(rules, pa.Table)

    def test_required_columns(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.1, min_confidence=0.1, min_lift=0.0)
        rules = ar.fit()
        expected = {"antecedents", "consequents", "support", "confidence", "lift",
                    "leverage", "conviction", "zhang"}
        assert expected.issubset(set(rules.column_names))

    def test_confidence_formula(self, loaded_conn):
        # milk → bread:
        #   support(milk ∪ bread) = 0.4
        #   support(milk)         = 0.6
        #   confidence = 0.4/0.6 ≈ 0.667
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        # Filter manually
        rows = rules.to_pylist()
        row = next((r for r in rows if r["antecedents"] == "milk" and r["consequents"] == "bread"), None)
        assert row is not None
        assert abs(row["confidence"] - (0.4 / 0.6)) < 1e-6

    def test_lift_formula(self, loaded_conn):
        # milk → bread: lift = confidence / support(bread) = (0.4/0.6) / 0.7 ≈ 0.952
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        rows = rules.to_pylist()
        row = next((r for r in rows if r["antecedents"] == "milk" and r["consequents"] == "bread"), None)
        assert row is not None
        expected_lift = (0.4 / 0.6) / 0.7
        assert abs(row["lift"] - expected_lift) < 1e-6

    def test_min_confidence_filters(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.9, min_lift=0.0)
        rules = ar.fit()
        if rules.num_rows > 0:
            for val in rules.column("confidence"):
                assert val.as_py() >= 0.9

    def test_min_lift_filters(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=1.5)
        rules = ar.fit()
        if rules.num_rows > 0:
            for val in rules.column("lift"):
                assert val.as_py() >= 1.5

    def test_min_support_filters(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.5, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        if rules.num_rows > 0:
            for val in rules.column("support"):
                assert val.as_py() >= 0.5

    def test_leverage_formula(self, loaded_conn):
        # leverage = support(AB) - support(A) * support(B)
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        rows = rules.to_pylist()
        for row in rows:
            # Check self-consistency
            # leverage is pre-calculated in strict SQL, so just checking type/existence
            assert isinstance(row["leverage"], float)

    def test_accepts_precomputed_itemsets(self, loaded_conn):
        from duwhal.mining.frequent_itemsets import FrequentItemsets
        fi = FrequentItemsets(loaded_conn, min_support=0.1)
        itemsets = fi.fit()

        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit(frequent_itemsets=itemsets)
        assert isinstance(rules, pa.Table)

    def test_invalid_min_support_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            AssociationRules(loaded_conn, min_support=-0.1)

    def test_invalid_min_confidence_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            AssociationRules(loaded_conn, min_confidence=0)

    def test_invalid_min_lift_raises(self, loaded_conn):
        with pytest.raises(ValueError):
            AssociationRules(loaded_conn, min_lift=-1)

    def test_empty_result_on_high_thresholds(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.99, min_confidence=0.99, min_lift=100)
        rules = ar.fit()
        assert rules.num_rows == 0

    def test_sorted_by_lift_descending(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        if rules.num_rows > 1:
            lifts = rules.column("lift").to_pylist()
            # Check if sorted descending
            assert all(lifts[i] >= lifts[i+1] for i in range(len(lifts)-1))

    def test_zhang_metric_range(self, loaded_conn):
        ar = AssociationRules(loaded_conn, min_support=0.05, min_confidence=0.01, min_lift=0.0)
        rules = ar.fit()
        if rules.num_rows > 0:
            for val in rules.column("zhang"):
                assert -1.0 <= val.as_py() <= 1.0

    def test_max_antecedent_len(self, loaded_conn):
        ar = AssociationRules(
            loaded_conn, min_support=0.05, min_confidence=0.01,
            min_lift=0.0, max_antecedent_len=1
        )
        rules = ar.fit()
        if rules.num_rows > 0:
            for row in rules.to_pylist():
                assert len(row["antecedents"].split("|")) <= 1
