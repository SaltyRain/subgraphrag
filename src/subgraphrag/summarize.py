# ---- verbalization of subgraph for semantic score ----
from collections import Counter
from typing import List, Tuple


def subgraph_to_stub(sg: dict, max_preds=6, max_nodes=8) -> str:
    """
    Build a compact, human-readable string that summarizes a subgraph.
    The stub is later used for shallow semantic scoring (e.g., BoW cosine)
    against the user question.

    Expected `sg` structure (minimal fields used here):
      sg["head"]:  str                # the head entity label for this subgraph
      sg["edges"]: List[Tuple[str,str,str]]  # list of (subject, predicate, object) labels
      sg["nodes"]: List[str]          # (not used directly here except to exclude head)

    Args:
        sg:        Subgraph dictionary produced upstream (labels, not QIDs).
        max_preds: How many top predicates to include (by frequency).
        max_nodes: How many top nodes to include (by degree, excluding the head).

    Returns:
        A short textual summary like:
        "Head: <HEAD>. Predicates: <p1, p2, ...>. Nodes: <n1, n2, ...>."
    """

    # Collect all predicate labels from edges; we summarize predicates
    # by their frequency to emphasize the most characteristic relations.
    preds = [p for _, p, _ in sg["edges"]]

    # Take the `max_preds` most frequent predicates (ties resolved by Counter's order).
    pred_top = ", ".join([p for p, _ in Counter(preds).most_common(max_preds)])

    # Head entity: we exclude it from the node list to avoid redundancy.
    head = sg["head"]

    # Compute a simple "degree" for each node as the number of times it
    # appears as subject or object. This favors hubs and entities central
    # to the subgraph's evidence.
    node_deg = Counter(
        [u for u, _, _ in sg["edges"]] +  # subjects
        [v for _, _, v in sg["edges"]]    # objects
    )

    # Select top nodes by degree, excluding the head; cap by `max_nodes`
    # to keep the stub short (token budget friendly).
    node_top = ", ".join(
        [n for n, _ in node_deg.most_common() if n != head][:max_nodes]
    )

    # Compose a compact summary string consumed by the BoW scorer.
    # The fixed template also reduces variance in tokenization.
    return f"Head: {head}. Predicates: {pred_top}. Nodes: {node_top}."



def path_to_stub(path: List[Tuple[str, str, str]], max_preds: int = 3, max_nodes: int = 3) -> str:
    """Compact textual summary of a single 1–2 hop path (used for BoW scoring)."""
    preds = [p for _, p, _ in path]
    pred_top = ", ".join([p for p, _ in Counter(preds).most_common(max_preds)])
    deg = Counter([u for u,_,_ in path] + [v for _,_,v in path])
    node_top = ", ".join([n for n, _ in deg.most_common(max_nodes)])
    return f"Predicates: {pred_top}. Nodes: {node_top}."