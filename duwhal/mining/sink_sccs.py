from __future__ import annotations
import pyarrow as pa
from typing import List, Dict, Set, Optional
from duwhal.core.connection import DuckDBConnection
import sys

# Increase recursion depth for deep graphs
sys.setrecursionlimit(100000)

class SinkSCCFinder:
    def __init__(self, conn: DuckDBConnection, table_name: str = "interactions", min_cooccurrence: int = 5):
        self.conn = conn
        self.table_name = table_name
        self.min_cooccurrence = min_cooccurrence

    def _build_adjacency(self, min_confidence: float) -> Dict[str, List[str]]:
        self.conn.execute("CREATE OR REPLACE TEMP TABLE _node_totals AS SELECT node_id, COUNT(*) AS total FROM {0} GROUP BY 1".format(self.table_name))
        self.conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _sink_edges AS 
            SELECT a.node_id AS source, b.node_id AS target, COUNT(*)::DOUBLE / t.total AS prob, COUNT(*) AS cooc
            FROM {self.table_name} a JOIN {self.table_name} b ON a.set_id = b.set_id AND a.node_id != b.node_id
            JOIN _node_totals t ON a.node_id = t.node_id
            GROUP BY 1, 2, t.total HAVING cooc >= {self.min_cooccurrence} AND prob >= {min_confidence}
        """)
        edges = self.conn.execute("SELECT source, target FROM _sink_edges").fetch_arrow_table().to_pylist()
        adj = {}
        for r in edges: adj.setdefault(r["source"], []).append(r["target"])
        return adj

    def _tarjan_scc(self, adj: Dict[str, List[str]]) -> List[List[str]]:
        state = {"idx": 0, "stack": [], "on_stack": set(), "index": {}, "lowlink": {}, "sccs": []}
        nodes = set(adj.keys()) | {t for targets in adj.values() for t in targets}
        
        def strongconnect(v):
            state["index"][v] = state["lowlink"][v] = state["idx"]
            state["idx"] += 1
            state["stack"].append(v); state["on_stack"].add(v)
            for w in adj.get(v, []):
                if w not in state["index"]:
                    strongconnect(w)
                    state["lowlink"][v] = min(state["lowlink"][v], state["lowlink"][w])
                elif w in state["on_stack"]:
                    state["lowlink"][v] = min(state["lowlink"][v], state["index"][w])
            if state["lowlink"][v] == state["index"][v]:
                scc = []
                while True:
                    w = state["stack"].pop(); state["on_stack"].remove(w); scc.append(w)
                    if w == v: break
                state["sccs"].append(scc)

        for n in nodes:
            if n not in state["index"]: strongconnect(n)
        return state["sccs"]

    def _identify_sinks(self, adj: Dict[str, List[str]], sccs: List[List[str]]) -> List[bool]:
        node_to_scc = {node: i for i, scc in enumerate(sccs) for node in scc}
        is_sink = [True] * len(sccs)
        for i, scc in enumerate(sccs):
            for node in scc:
                if any(node_to_scc.get(neighbor) != i for neighbor in adj.get(node, [])):
                    is_sink[i] = False; break
        return is_sink

    def find(self, min_confidence: float = 0.0) -> pa.Table:
        adj = self._build_adjacency(min_confidence)
        sccs = self._tarjan_scc(adj)
        is_sink = self._identify_sinks(adj, sccs)
        
        res = []
        for i, scc in enumerate(sccs):
            if is_sink[i]:
                mems = "|".join(sorted(scc))
                res.extend([{"node": n, "scc_id": i, "scc_size": len(scc), "is_sink": True, "members": mems} for n in scc])
        
        schema = pa.schema([("node", pa.string()), ("scc_id", pa.int32()), ("scc_size", pa.int32()), ("is_sink", pa.bool_()), ("members", pa.string())])
        return pa.Table.from_pylist(res, schema=schema) if res else pa.Table.from_pylist([], schema=schema)
