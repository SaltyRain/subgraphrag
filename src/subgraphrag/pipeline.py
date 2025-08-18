from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, cast
from tqdm import tqdm

from src.subgraphrag.compact import render_compact_chain, collapse_two_hop_paths
from src.subgraphrag.labeling import convert_paths_to_labels
from src.subgraphrag.prune import limit_explosive_predicates, prune_subgraph_by_top_paths
from src.subgraphrag.scoring import score_semantic_bow, score_seed_overlap, score_info, _minmax_norm
from src.subgraphrag.select import _sg_size
from src.subgraphrag.subgraph import Triplet, construct_two_hop_paths, build_head_subgraphs
from src.subgraphrag.text import _bow
from src.types import SRTKStructureWithTriplets, ContextGenerationStructure


@dataclass
class PreRankParams:
    # Parameters for pre-ranking subgraphs
    top_k: int = 8
    frac_cutoff: float = 0.70
    w_sem: float = 0.55
    w_seed: float = 0.20
    w_info: float = 0.25
    ## Pruning knobs
    explosive_preds: Optional[List[str]] = None
    cap_per_subject: int = 3
    top_paths_per_head: int = 6
    lam_fanout: float = 0.3

def construct_structured_subgraphs(
    labels_map: Dict[str, str],
    questions: List[SRTKStructureWithTriplets],
    params: PreRankParams
) -> List[ContextGenerationStructure]:
    """
        Converts triplets to tuples, builds 1–2 hop paths, verbalizes entities and builds Subgraph structures.
        Applies per-question normalization and combined scoring, then selects heads by fraction-of-best and top-k.
    """
    result = []

    # Unpack parameters
    top_k = params.top_k
    frac_cutoff = params.frac_cutoff
    w_sem = params.w_sem
    w_seed = params.w_seed
    w_info = params.w_info
    explosive_preds = params.explosive_preds or []
    cap_per_subject = params.cap_per_subject
    top_paths_per_head = params.top_paths_per_head
    lam_fanout = params.lam_fanout


    for question in tqdm(questions, desc="Constructing subgraphs"):
        triplets: List[Triplet] = [
            cast(Triplet, tuple(t)) for t in question["triplets"] if len(t) == 3
        ]
        paths = construct_two_hop_paths(triplets)
        labeled_paths = convert_paths_to_labels(paths, labels_map)

        # group labeled paths by head (subject of the first edge)
        paths_by_head: Dict[str, List[List[Tuple[str, str, str]]]] = {}
        for path in labeled_paths:
            if not path:
                continue
            head_node = path[0][0]
            paths_by_head.setdefault(head_node, []).append(path)

        subgraphs = build_head_subgraphs(
            labeled_paths,
            ignore_entities={"unknown"},
            dedup_paths=True, dedup_edges=True, deterministic=True,
        )

        sg_dict = {h: asdict(sg) for h, sg in subgraphs.items()}
        # Cache question BoW once per question
        q_bow = _bow(question["question"])

        # --- per-head pruning before scoring ---
        for head, sg in sg_dict.items():
            # (1) cap explosive predicates per subject
            if explosive_preds:
                sg = limit_explosive_predicates(
                    sg,
                    explosive_preds=explosive_preds,
                    cap_per_subject=cap_per_subject
                )
                sg_dict[head] = sg
            # (2) keep only edges that lie on top-scoring paths for this head
            head_paths = paths_by_head.get(head, [])
            sg = prune_subgraph_by_top_paths(
                sg,
                q_bow=q_bow,
                paths_for_head=head_paths,
                top_paths=top_paths_per_head,
                lam_fanout=lam_fanout
            )
            sg_dict[head] = sg



        # Map seed QIDs -> labels (fall back to QID string)
        seeds = question.get("question_entities") or []
        seed_labels = [labels_map.get(s, s) for s in seeds]

        # ----- compute raw component scores per head -----
        sem_raw: Dict[str, float] = {}
        seed_raw: Dict[str, float] = {}
        info_raw: Dict[str, float] = {}

        for head, sg in sg_dict.items():
            sem_raw[head] = score_semantic_bow(q_bow, sg)
            seed_raw[head] = score_seed_overlap(seed_labels, sg)
            info_raw[head] = score_info(sg)


        # ----- normalize components to [0,1] per question (scale alignment) -----
        sem_n  = _minmax_norm(sem_raw)
        seed_n = _minmax_norm(seed_raw)
        info_n = _minmax_norm(info_raw)

        # ----- combine with weights; store both raw and combined for auditability -----
        scores: Dict[str, float] = {}
        for head, sg in sg_dict.items():
            sem = sem_n.get(head, 0.0)
            seed = seed_n.get(head, 0.0)
            info = info_n.get(head, 0.0)
            combined = w_sem * sem + w_seed * seed + w_info * info
            scores[head] = combined

            # Persist diagnostics
            sg.setdefault("stats", {})
            sg["stats"]["rank_score"] = round(combined, 4)
            sg["stats"]["rank_components"] = {
                "sem_raw": round(sem_raw.get(head, 0.0), 4),
                "seed_raw": round(seed_raw.get(head, 0.0), 4),
                "info_raw": round(info_raw.get(head, 0.0), 4),
                "sem": round(sem, 4),
                "seed": round(seed, 4),
                "info": round(info, 4),
                "w_sem": w_sem, "w_seed": w_seed, "w_info": w_info,
            }

        # ----- fraction-of-best pruning + top-k with tie-break by size -----
        if scores:
            best = max(scores.values())
            kept_heads = [h for h in scores if scores[h] >= frac_cutoff * best]
            # tie-break: higher score first, then smaller subgraph
            kept_heads = sorted(
                kept_heads,
                key=lambda h: (-scores[h], _sg_size(sg_dict[h]))
            )[:top_k]
            sg_top = {h: sg_dict[h] for h in kept_heads}
        else:
            sg_top = {}

        result.append({
            "id": question["id"],
            "question": question["question"],
            "answers_str": question.get("answers_str", []),
            "subgraphs": sg_top
        })

    return result




def build_compact_contexts(
    questions: List[Dict[str, Any]],
    *,
    # number of heads (per question) to keep; None = keep all
    max_heads: Optional[int] = None,
    # path-collapsing limits
    max_o1: int = 4,
    max_p2: int = 2,
    max_o2: int = 3,
    dedup_selfloops: bool = True,
    # include collapsed bundles in the output (for debugging)
    include_bundles: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build compact, de-duplicated textual contexts for each question.

    Expected input per question:
      {
        "id": "...",
        "question": "...",
        "answers_str": [...],                  # optional
        "subgraphs": {
          "<HEAD>": {
            "head": "<HEAD>",
            "paths": [ [ (s,p,o), (s,p,o)? ], ... ],   # labels, not QIDs
            "stats": {"rank_score": ...}               # optional
          },
          ...
        }
      }

    Output per question:
      {
        "id": "...",
        "question": "...",
        "answers_str": [...],                 # passthrough if present
        "contexts": { "<HEAD>": "<compact text>", ... },
        # optional (if include_bundles=True)
        "bundles":  { "<HEAD>": [ {p1, objs, pivot, p2_bundles}, ... ], ... }
      }

    Head selection:
      - If `max_heads` is None, process all heads.
      - Otherwise sort heads by:
          1) stats.rank_score (descending) when available,
          2) number of paths (descending),
          3) head string (ascending),
        then keep the top `max_heads`.
    """
    outputs: List[Dict[str, Any]] = []

    for q in questions:
        qid = q.get("id")
        qtext = q.get("question", "")
        answers = q.get("answers_str", None)
        sg_map: Dict[str, Dict[str, Any]] = q.get("subgraphs", {}) or {}

        # sort heads with the stated priority
        def head_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, int, str]:
            head, sg = item
            stats = sg.get("stats") or {}
            score = stats.get("rank_score")
            score_val = float(score) if isinstance(score, (int, float)) else float("-inf")
            path_count = len(sg.get("paths") or [])
            # we want: score desc, path_count desc, head asc
            # implement by negating numeric fields so we can sort ascending
            return -score_val, -path_count, head or ""

        heads_sorted = sorted(sg_map.items(), key=head_sort_key)

        if max_heads is not None:
            heads_sorted = heads_sorted[:max_heads]

        contexts: Dict[str, str] = {}
        bundles_out: Dict[str, Any] = {}

        for head, sg in heads_sorted:
            raw_paths = sg.get("paths") or []

            # sanitize and coerce paths → List[List[Tuple[str,str,str]]]
            clean_paths: List[List[Tuple[str, str, str]]] = []
            for path in raw_paths:
                if not isinstance(path, list) or not path:
                    continue
                triples: List[Tuple[str, str, str]] = []
                valid = True
                for t in path:
                    if (
                        isinstance(t, (list, tuple)) and len(t) == 3
                        and all(isinstance(x, str) for x in t)
                    ):
                        triples.append((t[0], t[1], t[2]))
                    else:
                        valid = False
                        break
                if valid and triples:
                    clean_paths.append(triples)

            # collapse and render
            bundles = collapse_two_hop_paths(
                clean_paths,
                max_o1=max_o1,
                max_p2=max_p2,
                max_o2=max_o2,
                dedup_selfloops=dedup_selfloops,
            )
            compact_text = render_compact_chain(head, bundles)

            contexts[head] = compact_text
            if include_bundles:
                bundles_out[head] = bundles

        out: Dict[str, Any] = {
            "id": qid,
            "question": qtext,
            "contexts": contexts,
        }
        if answers is not None:
            out["answers_str"] = answers
        if include_bundles:
            out["bundles"] = bundles_out

        outputs.append(out)

    return outputs
