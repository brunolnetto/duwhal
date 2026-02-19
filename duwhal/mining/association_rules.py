from __future__ import annotations
import pyarrow as pa
from typing import Optional, List, Dict, Any
from duwhal.core.connection import DuckDBConnection
from duwhal.mining.frequent_itemsets import FrequentItemsets

class AssociationRules:
    METRICS = ["support", "confidence", "lift", "leverage", "conviction", "zhang"]

    def __init__(
        self,
        conn: DuckDBConnection,
        table_name: str = "interactions",
        min_support: float = 0.05,
        min_confidence: float = 0.1,
        min_lift: float = 0.0,
        max_antecedent_len: Optional[int] = None,
    ):
        if not (0 < min_support <= 1): raise ValueError("min_support must be in (0, 1]")
        if not (0 < min_confidence <= 1): raise ValueError("min_confidence must be in (0, 1]")
        if min_lift < 0: raise ValueError("min_lift must be non-negative")
        
        self.conn, self.table_name = conn, table_name
        self.min_support, self.min_confidence, self.min_lift = min_support, min_confidence, min_lift
        self.max_antecedent_len = max_antecedent_len
        self._rules, self.last_sql_ = None, ""

    def _compute_zhang(self, conf, supp_b):
        denom = max(conf * (1 - supp_b), supp_b * (1 - conf))
        return (conf - supp_b) / denom if denom > 0 else 0

    def _calculate_metrics(self, row: Dict[str, Any], support_b: float) -> Optional[Dict[str, Any]]:
        supp_a, supp_ab = row["support_a"], row["support"]
        conf = supp_ab / supp_a if supp_a > 0 else 0
        lift = conf / support_b if support_b > 0 else 0
        if conf < self.min_confidence or lift < self.min_lift: return None
        return {
            "antecedents": row["antecedents"], "consequents": row["consequents"],
            "support": supp_ab, "confidence": conf, "lift": lift,
            "leverage": supp_ab - (supp_a * support_b),
            "conviction": (1 - support_b) / (1 - conf + 1e-9),
            "zhang": self._compute_zhang(conf, support_b)
        }

    def _process_single_rule(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ants = set(row["antecedents"].split("|"))
        if self.max_antecedent_len is not None and len(ants) > self.max_antecedent_len: return None
        full = set(row["consequents_full"].split("|"))
        cons = "|".join(sorted(list(full - ants)))
        if not cons: return None
        supp_b = self.conn.execute(f"SELECT support FROM _fi WHERE itemset = '{cons}'").fetchone()
        if not supp_b: return None
        row["consequents"] = cons
        return self._calculate_metrics(row, supp_b[0])

    def _fetch_refined_rules(self):
        self.last_sql_ = "WITH rules AS (SELECT a.itemset AS antecedents, b.itemset AS consequents_full, a.support AS support_a, b.support AS support_ab FROM _fi a JOIN _fi b ON b.itemset LIKE a.itemset || '|%' OR b.itemset LIKE '%|' || a.itemset OR b.itemset LIKE '%|' || a.itemset || '|%' WHERE a.length < b.length) SELECT antecedents, consequents_full, support_ab AS support, support_a FROM rules"
        return self.conn.execute(self.last_sql_).fetch_arrow_table().to_pylist()

    def _get_empty_table(self):
        s = pa.schema([(m, pa.float64()) for m in self.METRICS] + [("antecedents", pa.string()), ("consequents", pa.string())])
        return pa.Table.from_pylist([], schema=s)

    def _fit_fi(self, fi):
        if fi is None: return FrequentItemsets(self.conn, table_name=self.table_name, min_support=self.min_support).fit()
        return fi

    def fit(self, frequent_itemsets: Optional[pa.Table] = None) -> pa.Table:
        fi = self._fit_fi(frequent_itemsets)
        self.conn.register("_fi", fi)
        processed = [p for p in (self._process_single_rule(row) for row in self._fetch_refined_rules()) if p]
        if not processed: return self._get_empty_table()
        processed.sort(key=lambda x: x["lift"], reverse=True)
        self._rules = pa.Table.from_pylist(processed)
        return self._rules
