import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Union, Optional

from dotenv import load_dotenv
import pandas as pd
from pandas import DataFrame

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import (
    # context_precision,
    # context_recall,
    # faithfulness,
    answer_relevancy,
    answer_correctness,
)

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from src.utils.fs import write_txt, write_df_to_csv


# ---------------------------
# IO helpers
# ---------------------------

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------
# RAGAS dataset conversion
# ---------------------------

def _pick_golds(r: Dict[str, Any], use_first_gold: bool) -> str:
    """
    Support both legacy 'answers_str' and new 'gold' fields.
    Returns a single reference string (possibly a comma-joined list).
    """
    golds: Optional[List[str]] = None
    if isinstance(r.get("answers_str"), list):
        golds = r["answers_str"]
    elif isinstance(r.get("gold"), list):
        golds = r["gold"]

    golds = golds or []
    if use_first_gold and golds:
        return str(golds[0])
    return ", ".join(map(str, golds)) if golds else ""


def to_single_turn_samples(
    rows: Iterable[Dict[str, Any]],
    *,
    use_first_gold: bool = False,
) -> List[SingleTurnSample]:
    """
    Convert records into Ragas SingleTurnSample list.

    Expected input fields per record (new format):
      - question: str
      - fused_context: str            # single, fused context
      - answer: str                   # predicted answer
      - gold OR answers_str: List[str] (gold(s))  [optional but preferred]
    """
    samples: List[SingleTurnSample] = []
    for r in rows:
        question = str(r.get("question", "")).strip()
        fused_context = str(r.get("fused_context", "")).strip()
        pred = str(r.get("answer", "")).strip()
        reference = _pick_golds(r, use_first_gold)

        # Skip malformed rows (Ragas expects non-empty question/response;
        # retrieved_contexts can be empty, но дадим 1 элемент для стабильности)
        if not question or not pred:
            continue

        contexts = [fused_context] if fused_context else []

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=pred,
                reference=reference,
            )
        )
    return samples


# ---------------------------
# Metrics factory
# ---------------------------

def build_metrics(include_relevancy: bool = False) -> List:
    """
    We deliberately disable context_* and faithfulness for this study.
    Primary focus: did the pipeline recover the correct answer?
    """
    m = [
        # NOTE: don't use them here, too expensive and not really relevant
        # context_precision,  # needs embeddings
        # context_recall,  # needs embeddings
        # faithfulness,  # needs LLM
        answer_correctness,  # LLM-based judging of correctness vs reference(s)
    ]
    if include_relevancy:
        # Optional; can be noisy, но иногда полезно для sanity-check
        m.append(answer_relevancy)
    return m


# ---------------------------
# LLM / Embeddings
# ---------------------------

def build_ollama_llm(
    model: str,
    base_url: str,
    temperature: float = 0.0,
    force_json: bool = False,  # kept for CLI compatibility, but ignored
) -> OllamaLLM:
    """
    Create an OllamaLLM for RAGAS judge. Force plain text; do NOT use JSON.
    """
    # Intentionally ignore force_json for RAGAS to avoid OutputParserException
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
        # explicit guard against JSON
        # system="Return plain text ONLY. Do not output JSON, keys, braces, code fences, or lists of statements."
    )


def build_ollama_embeddings(model: str, base_url: str) -> OllamaEmbeddings:
    # Not strictly needed if only correctness is used, but ragas API allows passing it.
    return OllamaEmbeddings(model=model, base_url=base_url)


# ---------------------------
# Evaluation runner
# ---------------------------

def run_evaluation(
    samples: List[SingleTurnSample],
    *,
    llm: OllamaLLM,
    embeddings: Optional[OllamaEmbeddings],
    metrics: List,
    batch_size: int = 8,
    show_progress: bool = True,
) -> EvaluationResult:
    dataset = EvaluationDataset(samples=samples)
    results: EvaluationResult = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        experiment_name='subgraphrag_evaluation',
        embeddings=embeddings,  # can be None if your metric set doesn’t need it
        show_progress=show_progress,
        batch_size=batch_size,
    )
    return results


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate final answers with Ragas (new fused format).")
    p.add_argument("--input_path", type=str, required=True, help="Path to JSONL with fused_context + answer (+ golds).")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save evaluation results.")
    p.add_argument("--host", type=str, default=None, help="Ollama base_url; overrides env LLM_BINDING_HOST.")
    p.add_argument("--llm_model", type=str, default=None, help="LLM model; overrides env LLM_MODEL_NAME.")
    p.add_argument("--embed_model", type=str, default=None, help="Embedding model; overrides env LLM_EMBEDDING_MODEL_NAME.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--include_relevancy", action="store_true", help="Also compute answer_relevancy (optional).")
    p.add_argument("--use_first_gold", action="store_true", help="Use only the first gold answer instead of comma-joined list.")
    p.add_argument("--force_json_mode", action="store_true", help="Try to force JSON mode on LLM (if supported).")
    p.add_argument("--no_embeddings", action="store_true", help="Skip embedding backend (okay if metrics don’t need it).")
    return p.parse_args()


def main():
    load_dotenv()

    args = parse_args()

    base_url = args.host or os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    llm_model = args.llm_model or os.getenv("LLM_MODEL_NAME", "llama3.1")
    embed_model = args.embed_model or os.getenv("LLM_EMBEDDING_MODEL_NAME", "nomic-embed-text")

    # Read input file
    rows = read_jsonl(args.input_path)

    # smoke test
    # rows = rows[:3]  # For quick testing, remove in production

    samples = to_single_turn_samples(rows, use_first_gold=args.use_first_gold)
    if not samples:
        raise RuntimeError("No valid samples found in the input file.")

    # Backends
    llm = build_ollama_llm(llm_model, base_url, temperature=0.0, force_json=args.force_json_mode)
    embeddings = None if args.no_embeddings else build_ollama_embeddings(embed_model, base_url)

    # Metrics (minimal set)
    metrics = build_metrics(include_relevancy=args.include_relevancy)

    # Evaluate
    results = run_evaluation(
        samples=samples,
        llm=llm,
        embeddings=embeddings,
        metrics=metrics,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    result_df: DataFrame = results.to_pandas()
    metrics_path = os.path.join(args.output_dir, "evaluation.metrics")
    write_txt(metrics_path, str(results))

    df_path = os.path.join(args.output_dir, "evaluation.csv")
    write_df_to_csv(df_path, result_df)

    print(f"Evaluation results saved to {metrics_path} and {df_path}")


if __name__ == "__main__":
    main()
