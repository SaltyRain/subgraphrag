from dataclasses import dataclass, asdict
from typing import (
    Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, Iterator, DefaultDict
)
from collections import defaultdict

Triplet = Tuple[str, str, str]
GraphPath = List[Triplet]
PathKey = Tuple[Triplet, ...]

def construct_two_hop_paths(
    triplets: Sequence[Triplet],
    seeds: Optional[Iterable[str]] = None,
    cap_per_node: Optional[int] = None,   # per-hop cap on explored outgoing edges
    include_one_hop: bool = True,         # include (s,p1,o1) paths
    dedup: bool = True,                   # deduplicate complete paths
    deterministic: bool = True,           # sort nodes/edges for reproducible order
    as_generator: bool = False,           # stream results instead of materializing
    predicate_allow: Optional[Iterable[str]] = None,  # whitelist of predicates
    predicate_deny: Optional[Iterable[str]] = None,   # blacklist of predicates
    max_paths_total: Optional[int] = None,            # global cap on emitted paths
) -> Union[List[List[Triplet]], Iterator[List[Triplet]]]:
    """
    Build 1-hop and 2-hop paths from RDF-like triplets.

    Each returned path is a list of 1 or 2 triplets:
      - 1-hop: (s, p1, o1)
      - 2-hop: (s, p1, o1), (o1, p2, o2)

    Notes:
      * cap_per_node applies independently at hop-1 and hop-2.
      * dedup removes duplicate complete paths (by exact triplet sequence).
      * deterministic=True sorts starts and outgoing edges for stable order.
      * predicate filters apply at both hops.
      * max_paths_total limits total yielded paths across all starts.
    """

    # Precompute predicate filters
    allow = set(predicate_allow) if predicate_allow else None
    deny = set(predicate_deny) if predicate_deny else None

    def predicate_ok(p: str) -> bool:
        if allow is not None and p not in allow:
            return False
        if deny is not None and p in deny:
            return False
        return True

    # Build adjacency list of outgoing edges: adj[s] -> [(p, o), ...]
    adj: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)

    # Remove duplicate triplets upfront to avoid duplicated work/output
    seen_triplets: Set[Triplet] = set()
    for s, p, o in triplets:
        t = (s, p, o)
        if t in seen_triplets:
            continue
        seen_triplets.add(t)
        if predicate_ok(p):
            adj[s].append((p, o))

    # Optionally sort outgoing edges for deterministic truncation by cap_per_node
    if deterministic:
        for s in adj:
            adj[s].sort()  # lexicographic by (p, o)

    # Determine start nodes: use seeds if provided and present in adj, else all with outgoing edges
    start_nodes: List[str]
    if seeds is not None:
        start_nodes = [s for s in seeds if s in adj]
    else:
        start_nodes = list(adj.keys())

    if deterministic:
        start_nodes.sort()

    def _iter_paths() -> Iterator[List[Triplet]]:
        # Tracks full paths to enforce deduplication across all starts
        seen_paths: Set[PathKey] = set()
        emitted = 0

        for s in start_nodes:
            outs1 = adj.get(s, [])
            if cap_per_node is not None:
                outs1 = outs1[:cap_per_node]

            # Hop-1 exploration
            for p1, o1 in outs1:
                path1: PathKey  = ((s, p1, o1),)

                # Optionally emit 1-hop path
                if include_one_hop:
                    if (not dedup) or (path1 not in seen_paths):
                        yield list(path1)
                        if dedup:
                            seen_paths.add(path1)
                        emitted += 1
                        if max_paths_total is not None and emitted >= max_paths_total:
                            return

                # Hop-2 exploration from o1
                outs2 = adj.get(o1, [])
                if cap_per_node is not None:
                    outs2 = outs2[:cap_per_node]

                for p2, o2 in outs2:
                    # Cut trivial loops: avoid returning to s or staying on o1
                    if o2 == s or o2 == o1:
                        continue

                    path2: PathKey  = ((s, p1, o1), (o1, p2, o2))
                    if (not dedup) or (path2 not in seen_paths):
                        yield [*path2]
                        if dedup:
                            seen_paths.add(path2)
                        emitted += 1
                        if max_paths_total is not None and emitted >= max_paths_total:
                            return

    return _iter_paths() if as_generator else list(_iter_paths())


@dataclass
class Subgraph:
    """Container for a head-centric subgraph."""
    head: str
    nodes: List[str]           # unique, sorted
    edges: List[Triplet]       # unique, sorted lexicographically
    paths: List[GraphPath]          # original (deduplicated) paths that define this subgraph
    stats: Dict[str, int]      # convenience counters (n_nodes, n_edges, n_paths)


def build_head_subgraphs(
    paths: Sequence[GraphPath],
    *,
    dedup_paths: bool = True,                  # deduplicate identical paths
    dedup_edges: bool = True,                  # deduplicate identical triplets
    deterministic: bool = True,                # sort nodes/edges/paths for stable output
    ignore_entities: Optional[Iterable[str]] = None,   # e.g., {"unknown"}
    predicate_allow: Optional[Iterable[str]] = None,    # whitelist predicates
    predicate_deny: Optional[Iterable[str]] = None,     # blacklist predicates
    max_paths_per_head: Optional[int] = None,  # cap number of kept paths per head
    max_edges_per_head: Optional[int] = None,  # cap number of kept edges per head (after dedup/filter)
) -> Dict[str, Subgraph]:
    """
    Group 1/2-hop paths into head-centric subgraphs.
    A "head" is the subject of the first triplet in each path.
    """
    ignore_set: Set[str] = set(ignore_entities or [])
    allow_set: Optional[Set[str]] = set(predicate_allow) if predicate_allow else None
    deny_set: Optional[Set[str]] = set(predicate_deny) if predicate_deny else None

    def pred_ok(p: str) -> bool:
        if allow_set is not None and p not in allow_set:
            return False
        if deny_set is not None and p in deny_set:
            return False
        return True

    # 1) Bucket paths by head entity (subject of the first triplet)
    buckets: Dict[str, List[GraphPath]] = defaultdict(list)

    if dedup_paths:
        # Use a set of tuple-ized paths for deduplication
        seen_path_keys: Set[Tuple[Triplet, ...]] = set()
        for path in paths:
            if not path:
                continue
            head = path[0][0]
            if head in ignore_set:
                continue
            # All path triplets must pass predicate filter
            filtered = []
            ok = True
            for (s, p, o) in path:
                if s in ignore_set or o in ignore_set or (not pred_ok(p)):
                    ok = False
                    break
                filtered.append((s, p, o))
            if not ok:
                continue
            key = tuple(filtered)
            if key in seen_path_keys:
                continue
            seen_path_keys.add(key)
            buckets[head].append(list(key))
    else:
        for path in paths:
            if not path:
                continue
            head = path[0][0]
            if head in ignore_set:
                continue
            # Keep only if every triplet passes filters
            keep = True
            for (s, p, o) in path:
                if s in ignore_set or o in ignore_set or (not pred_ok(p)):
                    keep = False
                    break
            if keep:
                buckets[head].append(path)

    # 2) Optionally cap number of paths per head (preserving insertion order)
    if max_paths_per_head is not None:
        for h in list(buckets.keys()):
            buckets[h] = buckets[h][:max_paths_per_head]

    # 3) For each head, build nodes/edges sets from its paths
    result: Dict[str, Subgraph] = {}
    for head, head_paths in buckets.items():
        edge_set: Set[Triplet] = set()
        node_set: Set[str] = set([head])

        for path in head_paths:
            for (s, p, o) in path:
                if not pred_ok(p):
                    continue
                if s in ignore_set or o in ignore_set:
                    continue
                edge_set.add((s, p, o))
                node_set.add(s)
                node_set.add(o)

        # 4) Optional cap on edges (after dedup/filter)
        edges: List[Triplet]
        if deterministic:
            edges = sorted(edge_set)
        else:
            edges = list(edge_set)

        if max_edges_per_head is not None and len(edges) > max_edges_per_head:
            edges = edges[:max_edges_per_head]

        # Recompute nodes from kept edges to ensure consistency after edge cap
        if edges:
            node_set = set()
            for (s, _, o) in edges:
                node_set.add(s)
                node_set.add(o)
            node_set.add(head)

        nodes = sorted(node_set) if deterministic else list(node_set)

        # Sort paths deterministically (by lexicographic triplet tuples) if requested
        out_paths: List[GraphPath]
        if deterministic:
            out_paths = sorted(
                head_paths,
                key=lambda path: tuple(path)
            )
        else:
            out_paths = list(head_paths)

        subgraph = Subgraph(
            head=head,
            nodes=nodes,
            edges=edges,
            paths=out_paths,
            stats={
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "n_paths": len(out_paths),
            },
        )
        result[head] = subgraph

    return result


# --- Example usage ---
# grouped = build_head_subgraphs(paths, ignore_entities={"unknown"}, predicate_deny={"shares border with"})
# for head, sg in grouped.items():
#     print(head, sg.stats, sg.nodes[:5], sg.edges[:3])
