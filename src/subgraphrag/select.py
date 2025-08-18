def _sg_size(sg: dict) -> int:
    """Tie-breaker: prefer smaller subgraphs by number of edges (triples)."""
    return len(sg.get("edges", ()))