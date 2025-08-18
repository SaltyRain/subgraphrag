from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, DefaultDict


Triple = Tuple[str, str, str]  # (subject, predicate, object)
Path2Hop = List[Triple]        # 1–2 triples (we assume max length = 2)


def collapse_two_hop_paths(
    paths: List["Path2Hop"],
    *,
    max_o1: int = 4,      # maximum number of 1-hop objects to retain per p1
    max_p2: int = 2,      # maximum number of distinct second-hop predicates to expose
    max_o2: int = 3,      # maximum objects per second-hop predicate group
    dedup_selfloops: bool = True
) -> List[Dict[str, Any]]:
    """
    Collapse a set of 1–2 hop paths for a *single head* entity into compact bundles.

    Input
    -----
    paths : List[Path2Hop]
        Each path is a list with 1 or 2 triples in label space:
            path[0] = (head, p1, o1)
            path[1] = (o1, p2, o2)      # optional
        All paths are assumed to share the same head (subject of the first triple).

    Output
    ------
    List[Dict]
        A list of "bundles" (one per distinct p1) sorted by estimated strength:
        [
          {
            "p1": str,                 # first-hop predicate
            "objs": [o1, ...],         # top-1hop objects for p1 (up to max_o1)
            "pivot": o1*,              # the o1 chosen to expose its second-hop neighborhood
            "p2_bundles": [            # second-hop groups (from the pivot only)
              {"p2": str, "objs": [o2, ...]},   # up to max_o2 per p2
              ...
            ]
          },
          ...
        ]

    Notes
    -----
    - The goal is a compact, non-redundant view that keeps first-hop coverage
      while showing *representative* second-hop details via a pivot object.
    - We count frequencies of o1 and (p2, o2) to rank items deterministically.
    - Self-loops (head == o1) can be optionally removed.
    """
    if not paths:
        return []

    # (p1) -> Counter(o1) : 1-hop popularity (no head in the key)
    o1_counts: DefaultDict[str, Counter] = defaultdict(Counter)

    # (p1, o1) -> list of (p2, o2) instances : raw second-hop neighborhood
    cont2: DefaultDict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    for path in paths:
        if not path:
            continue

        # First hop
        s1, p1, o1 = path[0]
        if dedup_selfloops and s1 == o1:
            # Skip degenerate self-loops like (head, p, head)
            continue

        o1_counts[p1][o1] += 1

        # Optional second hop
        if len(path) > 1:
            _, p2, o2 = path[1]
            # Attach the 2-hop continuation to this (p1, o1)
            cont2[(p1, o1)].append((p2, o2))

    bundles: List[Dict[str, Any]] = []

    # Build bundles per distinct p1
    for p1, cnt in o1_counts.items():
        # Deterministic top-k of o1: by frequency desc, then lexical asc
        o1_freq_sorted = cnt.most_common()
        o1_freq_sorted.sort(key=lambda kv: (-kv[1], kv[0]))
        o1_top = [o for o, _ in o1_freq_sorted[:max_o1]]

        # Pick a pivot: the o1 exposing the richest 2-hop neighborhood
        # Deterministic tie-break: lexical asc
        pivot = None
        if o1_top:
            o1_scored = sorted(
                o1_top,
                key=lambda o: (-len(cont2.get((p1, o), [])), o)
            )
            pivot = o1_scored[0]

        # Group second-hop neighborhood for the pivot: p2 -> Counter(o2)
        p2_bundles: List[Dict[str, Any]] = []
        if pivot is not None:
            pairs = cont2.get((p1, pivot), [])
            by_p2: DefaultDict[str, Counter] = defaultdict(Counter)
            for p2, o2 in pairs:
                by_p2[p2][o2] += 1

            # For each p2, take top-o2 by frequency desc, then lexical asc
            p2_groups: List[Dict[str, Any]] = []
            for p2, o2cnt in by_p2.items():
                o2_freq_sorted = o2cnt.most_common()
                o2_freq_sorted.sort(key=lambda kv: (-kv[1], kv[0]))
                objs = [o for o, _ in o2_freq_sorted[:max_o2]]
                p2_groups.append({"p2": p2, "objs": objs})

            # Keep strongest p2 groups (by number of objs), tie-break by p2 lexically
            p2_groups.sort(key=lambda g: (-len(g["objs"]), g["p2"]))
            p2_bundles = p2_groups[:max_p2]

        bundles.append({
            "p1": p1,
            "objs": o1_top,
            "pivot": pivot,
            "p2_bundles": p2_bundles
        })

    # Sort bundles by "strength": more o1 plus more exposed o2 from the pivot
    def bundle_strength(b: Dict[str, Any]) -> int:
        return len(b["objs"]) + sum(len(g["objs"]) for g in b["p2_bundles"])

    bundles.sort(key=bundle_strength, reverse=True)
    return bundles


def render_compact_chain(head: str, bundles: List[Dict[str, Any]]) -> str:
    """
    Render a compact, human-readable chain for LLM consumption.

    Format
    ------
    [HEAD] <head>
    <head> — <p1> — {o1; o1; ...}
      ⟶ [from pivot: <o1*>]
         <p2> — {o2; o2; ...}
         <p2> — {o2; o2; ...}
    ...

    Notes
    -----
    - We keep the format intentionally simple and repetitive to help small LLMs
      parse the structure reliably.
    - Unicode dashes/arrows are used for readability; replace with ASCII if needed.
    """
    lines = [f"[HEAD] {head}"]
    for b in bundles:
        o1 = "; ".join(b.get("objs", []))
        lines.append(f"{head} — {b['p1']} — {{{o1}}}")
        pivot = b.get("pivot")
        p2_bundles = b.get("p2_bundles") or []
        if pivot and p2_bundles:
            lines.append(f"  ⟶ [from pivot: {pivot}]")
            for g in p2_bundles:
                o2 = "; ".join(g.get("objs", []))
                lines.append(f"     {g['p2']} — {{{o2}}}")
    return "\n".join(lines)
