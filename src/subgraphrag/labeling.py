# --- convert QID -> label -----------------------------------------------
from typing import Tuple, Dict, List

from src.subgraphrag.subgraph import Triplet


def triplet_to_labels(
    t: Tuple[str, str, str],
    labels_map: Dict[str, str],
    unknown: str = "unknown", # NOTE: not used, but kept for later research
) -> Tuple[str, str, str]:
    s, p, o = t

    def lab(x):
        # Fallback to QID string instead of dropping the node
        return labels_map.get(x, x)

    return lab(s), lab(p), lab(o)

def convert_paths_to_labels(
    paths: List[List[Triplet]],
    labels_map: Dict[str, str],
    unknown: str = "unknown",
) -> List[List[Tuple[str, str, str]]]:
    return [
        [triplet_to_labels(t, labels_map, unknown) for t in path]
        for path in paths
    ]