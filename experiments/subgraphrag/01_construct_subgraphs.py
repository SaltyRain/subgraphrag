import argparse
import json
from typing import Dict, List
from pathlib import Path

from src.subgraphrag.pipeline import construct_structured_subgraphs, PreRankParams, build_compact_contexts
from src.types import SRTKStructureWithTriplets


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _derive_structured_path(output_path: Path) -> Path:
    """
    If user didn't pass --structured_path, derive it from --output_path.
    e.g., results/final.jsonl -> results/final.structured.jsonl
    """
    base = output_path.with_suffix("")  # drop .json/.jsonl/etc.
    return base.with_name(base.name + ".structured.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Build structured subgraphs (debug) and compact contexts (final)."
    )

    # I/O
    parser.add_argument(
        "--input_path", type=Path, required=True,
        help="Path to input JSONL with questions + retrieved triplets."
    )
    parser.add_argument(
        "--labels_map_path", type=Path, required=True,
        help="Path to JSON mapping QID -> label."
    )
    parser.add_argument(
        "--output_path", type=Path, required=True,
        help="Path to save FINAL compact contexts (JSONL)."
    )
    parser.add_argument(
        "--structured_path", type=Path, default=None,
        help="Optional path to save INTERMEDIATE structured subgraphs (JSONL). "
             "If omitted, will derive from --output_path."
    )

    # Pre-reranking (structured stage) hyperparams
    parser.add_argument("--top-k", type=int, default=8,
                        help="Max heads to keep after pre-reranking (default: 8).")
    parser.add_argument("--frac-cutoff", type=float, default=0.70,
                        help="Keep heads with score >= cutoff * best_score (default: 0.70).")
    parser.add_argument("--w-sem", type=float, default=0.55,
                        help="Weight for normalized semantic score (default: 0.55).")
    parser.add_argument("--w-seed", type=float, default=0.20,
                        help="Weight for normalized seed-overlap score (default: 0.20).")
    parser.add_argument("--w-info", type=float, default=0.25,
                        help="Weight for normalized info score (default: 0.25).")

    # Pruning knobs (structured stage)
    parser.add_argument("--explosive-preds", nargs="*", default=[
        "contains the administrative territorial entity",
        "has part"
    ], help="Predicate labels to cap fan-out for (case-insensitive).")
    parser.add_argument("--cap-per-subject", type=int, default=3,
                        help="Max edges per (subject, explosive_predicate) (default: 3).")
    parser.add_argument("--top-paths-per-head", type=int, default=6,
                        help="Keep edges only from top-N paths per head (default: 6).")
    parser.add_argument("--lam-fanout", type=float, default=0.3,
                        help="Penalty weight for fan-out in path scoring (default: 0.3).")

    # Compacting (final stage) hyperparams
    parser.add_argument("--compact-max-heads", type=int, default=None,
                        help="Max heads per question to include in final contexts. "
                             "None = keep all heads selected in structured stage.")
    parser.add_argument("--max-o1", type=int, default=4,
                        help="Max number of 1-hop objects in a p1 bundle (default: 4).")
    parser.add_argument("--max-p2", type=int, default=2,
                        help="Max number of distinct p2 bundles from pivot (default: 2).")
    parser.add_argument("--max-o2", type=int, default=3,
                        help="Max number of objects per p2 bundle (default: 3).")
    parser.add_argument("--no-dedup-selfloops", action="store_true",
                        help="Disable self-loop deduplication in path collapsing.")

    args = parser.parse_args()

    # ---- Load inputs
    with open(args.labels_map_path, 'r', encoding='utf-8') as f:
        labels_map: Dict[str, str] = json.load(f)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        questions: List[SRTKStructureWithTriplets] = [json.loads(line) for line in f]

    # ---- Stage 1: structured subgraphs (with pre-reranking/pruning)
    params = PreRankParams(
        top_k=args.top_k,
        frac_cutoff=float(args.frac_cutoff),
        w_sem=float(args.w_sem),
        w_seed=float(args.w_seed),
        w_info=float(args.w_info),
        explosive_preds=[p.lower() for p in (args.explosive_preds or [])],
        cap_per_subject=int(args.cap_per_subject),
        top_paths_per_head=int(args.top_paths_per_head),
        lam_fanout=float(args.lam_fanout),
    )

    structured = construct_structured_subgraphs(
        labels_map=labels_map,
        questions=questions,
        params=params,
    )

    # Decide intermediate path
    structured_path: Path = args.structured_path or _derive_structured_path(args.output_path)
    _ensure_parent(structured_path)
    with open(structured_path, 'w', encoding='utf-8') as f:
        for entry in structured:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Stage 1] Structured subgraphs saved to {structured_path}")

    # ---- Stage 2: compact contexts
    dedup_selfloops = not args.no_dedup_selfloops

    compact = build_compact_contexts(
        structured,
        max_heads=args.compact_max_heads,
        max_o1=args.max_o1,
        max_p2=args.max_p2,
        max_o2=args.max_o2,
        dedup_selfloops=dedup_selfloops,
        include_bundles=False,  # flip to True if you want to debug bundles
    )

    _ensure_parent(args.output_path)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for entry in compact:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Stage 2] Compact contexts saved to {args.output_path}")


if __name__ == "__main__":
    main()
