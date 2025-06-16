from __future__ import annotations
import re, numpy as np, torch
from torch_geometric.data import Data
from modules.op_maps import NODE_TYPE_MAP, EDGE_TYPE_MAP

# ───────────────────────── helpers ─────────────────────────
_PAT_OP_FALLBACK = re.compile(r"(scan|join|agg|sort|hash|seek|filter|index)", re.I)
_WS_UNDERSCORE   = re.compile(r"[\s_]+")

def canonical(op: str) -> str:
    return _WS_UNDERSCORE.sub("", op.lower())

def _unwrap_plan(root: dict) -> dict:                   # unchanged
    if "Plan" in root and isinstance(root["Plan"], dict):
        return root["Plan"]
    if "plan" in root:
        def dfs(obj):
            if isinstance(obj, dict):
                if "Plan" in obj:
                    return obj["Plan"]
                for v in obj.values():
                    r = dfs(v)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for elt in obj:
                    r = dfs(elt)
                    if r is not None:
                        return r
            return None
        hit = dfs(root["plan"])
        if hit is not None:
            return hit
    return root                                          # fallback

def _raw_op(node: dict) -> str:                          # unchanged
    for k, v in node.items():
        if "node" in k.lower() and "type" in k.lower():
            return str(v)
        if isinstance(v, str) and _PAT_OP_FALLBACK.search(v):
            return v
    return "Unknown"

def _find_num(node: dict, contains: list[str]) -> float: # unchanged
    for k, v in node.items():
        if isinstance(v, (int, float)) and all(tok in k.lower() for tok in contains):
            return float(v)
    return 0.0

def _fanout(node: dict) -> float:                        # unchanged
    for v in node.values():
        if isinstance(v, list):
            return float(sum(1 for elt in v if isinstance(elt, dict)))
    return 0.0

# ───────── feature builders ─────────
def _node_feats(node: dict) -> np.ndarray:               # unchanged
    op_can = canonical(_raw_op(node))
    NODE_TYPE_MAP.setdefault(op_can, len(NODE_TYPE_MAP))
    op_id  = NODE_TYPE_MAP[op_can]

    est   = np.log1p(_find_num(node, ["row"]))
    width = np.log1p(_find_num(node, ["width"]))
    scost = np.log1p(_find_num(node, ["startup", "cost"]))
    tcost = np.log1p(_find_num(node, ["total",   "cost"]))
    fan   = _fanout(node)
    return np.array([op_id, est, width, scost, tcost, fan], dtype=np.float32)

def _edge_type(child: dict) -> int:                      # unchanged
    jt = child.get("Join Type")
    if jt is None:
        for k, v in child.items():
            if isinstance(v, str) and "join" in k.lower():
                jt = v
                break
    jt = jt or "child"
    jt_can = canonical(jt)
    EDGE_TYPE_MAP.setdefault(jt_can, len(EDGE_TYPE_MAP))
    return EDGE_TYPE_MAP[jt_can]

# ─────────── main public entry ───────────
def parse_plan(raw_json: dict) -> Data:
    """
    Parameters
    ----------
    raw_json : dict
        Either the whole EXPLAIN wrapper or already the inner “Plan”.

    Returns
    -------
    Data with .x, .edge_index, .edge_attr, .sql_ids, .sql_mask, .y
    """
    root = _unwrap_plan(raw_json)

    feats, ei, ea = [], [], []

    def walk(node: dict, parent: int | None = None):
        idx = len(feats)
        feats.append(_node_feats(node))
        if parent is not None:
            ei.append((parent, idx))
            ea.append(_edge_type(node))
        for v in node.values():
            if isinstance(v, list):
                for elt in v:
                    if isinstance(elt, dict):
                        walk(elt, idx)
    walk(root)

    x  = torch.tensor(np.stack(feats), dtype=torch.float32)
    ei_t = torch.tensor(ei, dtype=torch.long).t() if ei else torch.empty((2,0), dtype=torch.long)
    ea_t = torch.tensor(ea, dtype=torch.long)     if ea else torch.empty((0,),  dtype=torch.long)

    # ► Dummy SQL tensors (BERT etc. can plug in later)
    sql_ids  = torch.zeros((1,), dtype=torch.long)  # shape [L] (L=1)
    sql_mask = torch.ones_like(sql_ids)

    # ► Latency target from runtime_list
    runtimes = raw_json.get("runtime_list", [])
    lat = float(sum(runtimes)/len(runtimes)) if runtimes else 0.0
    y   = torch.tensor([lat], dtype=torch.float32)

    return Data(x=x, edge_index=ei_t, edge_attr=ea_t,
                sql_ids=sql_ids, sql_mask=sql_mask, y=y)
