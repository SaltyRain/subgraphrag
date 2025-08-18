from collections import Counter
from typing import Dict

from src.subgraphrag.summarize import subgraph_to_stub
from src.subgraphrag.text import _bow, _cosine_bow


# Pre-reranking of heads
def score_semantic_bow(q_bow: Counter, sg: dict) -> float:
    """Semantics via BoW cosine: cache question BoW outside and pass here."""
    return _cosine_bow(q_bow, _bow(subgraph_to_stub(sg)))

def score_seed_overlap(seeds_labels: list[str], sg: dict) -> float:
    """Seed overlap: seeds_labels is a list of labels (QIDs) to check against."""
    if not seeds_labels:
        return 0.0
    nodes = set(sg["nodes"])
    inter = nodes.intersection(seeds_labels)
    return len(inter) / max(1, len(nodes))

def score_info(sg: dict) -> float:
    edges = sg["edges"]; nodes = sg["nodes"]
    if not edges or not nodes:
        return 0.0
    diversity = len({p for _, p, _ in edges}) / len(edges) # distinct predicates ratio
    density = min(1.0, len(edges) / len(nodes)) # edges per node (capped)
    return 0.5*diversity + 0.5*density


def _minmax_norm(d: Dict[str, float]) -> Dict[str, float]:
    """Per-question min–max normalization to [0,1]. Keeps constants unchanged."""
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        # All equal → return zeros (or ones). Using zeros keeps weights' effect explicit.
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}