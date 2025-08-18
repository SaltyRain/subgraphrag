# --- paths pruning --------------------------------------------
from collections import Counter
from typing import Dict, Tuple, List
import math

from src.subgraphrag.summarize import path_to_stub
from src.subgraphrag.text import _cosine_bow, _bow


def limit_explosive_predicates(sg: dict, explosive_preds: List[str], cap_per_subject: int = 3) -> dict:
    """
    Cap fan-out for a set of 'explosive' predicate labels per subject.
    Keeps at most `cap_per_subject` edges (s, p, *) for each (subject, predicate).
    Predicate matching is case-insensitive over labels.
    """
    if not sg.get("edges"):
        return sg
    exp = {p.lower() for p in (explosive_preds or [])}
    buckets: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
    for (s, p, o) in sg["edges"]:
        key = (s, p.lower())
        buckets.setdefault(key, []).append((s, p, o))
    new_edges = []
    for (s, p_lc), triples in buckets.items():
        if p_lc in exp:
            new_edges.extend(triples[:cap_per_subject])  # cap
        else:
            new_edges.extend(triples)
    # rebuild nodes
    nodes = set()
    for (u, _, v) in new_edges:
        nodes.add(u); nodes.add(v)
    nodes.add(sg["head"])
    sg["edges"] = new_edges
    sg["nodes"] = list(nodes)
    return sg



def _fanout_counts(sg: dict) -> Dict[Tuple[str, str], int]:
    """Count fan-out per (subject, predicate) within subgraph (labels-based)."""
    c = Counter()
    for (u, p, v) in sg.get("edges", []):
        c[(u, p)] += 1
    return c

def score_path_bow(q_bow: Counter, path: List[Tuple[str, str, str]], fanout: Dict[Tuple[str, str], int], lam: float = 0.3) -> float:
    """
    BoW cosine between question and a path stub, penalized by fan-out.
    penalty = average(log1p(fanout(s,p))) over edges in the path, scaled by `lam`.
    """
    sem = _cosine_bow(q_bow, _bow(path_to_stub(path)))
    if not path:
        return sem
    pen = sum(math.log1p(fanout.get((u, p), 0)) for (u, p, v) in path) / len(path)
    return sem - lam * pen


def prune_subgraph_by_top_paths(
    sg: dict,
    q_bow: Counter,
    paths_for_head: List[List[Tuple[str, str, str]]],
    top_paths: int = 6,
    lam_fanout: float = 0.3
) -> dict:
    """
    Keep only edges that belong to the top-scoring paths for this head.
    If pruning removes everything, fall back to original sg.
    """
    if not paths_for_head:
        return sg
    fanout = _fanout_counts(sg)
    # Path-level pruning inside a head
    scored = [(score_path_bow(q_bow, path, fanout, lam=lam_fanout), path) for path in paths_for_head]
    scored.sort(key=lambda x: x[0], reverse=True)
    keep: set = set()
    for _, path in scored[:top_paths]:
        for e in path:
            keep.add(e)
    new_edges = [e for e in sg.get("edges", []) if e in keep]
    if not new_edges:  # fallback
        return sg
    nodes = set()
    for (u, _, v) in new_edges:
        nodes.add(u); nodes.add(v)
    nodes.add(sg["head"])
    sg["edges"] = new_edges
    sg["nodes"] = list(nodes)
    return sg