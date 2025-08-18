import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Union

from dotenv import load_dotenv
from lightrag import QueryParam
from tqdm import tqdm

from src.lightrag.initialize_rag import initialize_rag
from src.prompts.answer_prompt import SYSTEM_ANSWER, ANSWER_PROMPT, ABSTAIN_TOKEN
from src.utils.fs import write_txt, write_df_to_csv
from src.utils.logger import configure_logger

# RAGAS
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    # context_precision,
    # context_recall,
    # faithfulness,
    # answer_relevancy,
    answer_correctness,
)

# Ollama backends for RAGAS
from langchain_ollama import OllamaLLM, OllamaEmbeddings


# -------------------------- CLI -------------------------- #

def add_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightRAG: generate answers and evaluate with RAGAS (with contexts)."
    )

    # IO paths
    parser.add_argument(
        "--storage_dir",
        type=Path,
        default=Path("../../storage/lightrag"),
        help="Directory with LightRAG storage/index."
    )
    parser.add_argument(
        "--questions_path",
        type=Path,
        default=Path("../../outputs/processed/test.jsonl"),
        help="JSONL with fields: id, question, answers_str (list of golds)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../../outputs/lightrag/results"),
        help="Where to save answers and evaluations."
    )
    parser.add_argument(
        "--answers_filename",
        type=str,
        default="answers.jsonl",
        help="Filename for generated answers (inside output_dir)."
    )
    parser.add_argument(
        "--eval_filename",
        type=str,
        default="ragas_eval.json",
        help="Filename for RAGAS scores (inside output_dir)."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("../../logs/lightrag/pipeline"),
        help="Directory for timestamped logs."
    )

    # Pipeline toggles
    parser.add_argument(
        "--generate_answers",
        action="store_true",
        help="Run answer generation. If not set and answers file is missing, it will be created."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAGAS evaluation on produced answers (requires contexts)."
    )

    # LightRAG query mode
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        help="LightRAG query mode."
    )

    # RAGAS params
    parser.add_argument(
        "--ragas_batch_size",
        type=int,
        default=8,
        help="Batch size for RAGAS."
    )

    # Judge LLM (Ollama) and embeddings for RAGAS
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Ollama model name used as judge (e.g., 'llama3.1'). If not set, RAGAS will use its default."
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name for RAGAS (e.g., 'nomic-embed-text')."
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=None,
        help="Base URL for Ollama (e.g., http://localhost:11434). If not set, uses env or library default."
    )

    return parser.parse_args()


# ---------------------- IO helpers ---------------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ------------- Answer generation (async, with contexts) ------------- #

async def generate_answers_lightrag(
    questions: List[Dict[str, Any]],
    storage_dir: Path,
    output_dir: Path,
    answers_filename: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"],
    logger
) -> Path:
    """
    Generate answers and persist both the answer and retrieved contexts per question.
    Writes incrementally after each question.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / answers_filename

    # debug for first 2 questions
    # questions = questions[:20]

    rag = await initialize_rag(working_dir=storage_dir)

    await rag.aclear_cache(modes=[mode])

    # Load already processed IDs
    processed_ids: Set[str] = set()
    if answers_path.exists():
        for line in iter_jsonl(answers_path):
            _id = line.get("id")
            if _id:
                processed_ids.add(_id)

    unanswered = [q for q in questions if q.get("id") not in processed_ids]
    logger.info(f"Unanswered: {len(unanswered)} / {len(questions)}")

    if not unanswered:
        logger.info("All questions already answered. Skipping generation.")
        return answers_path

    with answers_path.open("a", encoding="utf-8") as f, tqdm(
        total=len(unanswered), desc="🧠 Generating answers"
    ) as pbar:
        for q in unanswered:
            qid = q.get("id")
            qtext = q.get("question", "")
            if not qid or not qtext:
                logger.warning("Skipping malformed question row.")
                pbar.update(1)
                continue
            try:
                resp = await rag.aquery(
                    query=qtext,
                    param=QueryParam(
                        mode=mode,
                        response_type='Single Line',
                        user_prompt=ANSWER_PROMPT.format(
                            question="{query}",
                            context="{context_data}",
                            abstain=ABSTAIN_TOKEN,
                            max_token_for_local_context=2048,
                            max_token_for_global_context=2048,
                        ),
                    ),
                    system_prompt=SYSTEM_ANSWER,
                )

                # Answer: try string or dict["answer"]
                if isinstance(resp, str):
                    answer = resp
                elif isinstance(resp, dict) and isinstance(resp.get("answer"), str):
                    answer = resp["answer"]
                else:
                    answer = str(resp)


                out = {
                    "id": qid,
                    "question": qtext,
                    "answer": answer,
                    "gold": q.get("answers_str", []),
                }
            except Exception as e:
                logger.error(f"Error on ID={qid}: {e}")
                out = {
                    "id": qid,
                    "question": qtext,
                    "answer": f"ERROR: {e}",
                    "gold": q.get("answers_str", []),
                }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)

    logger.info(f"✅ Answers written to {answers_path}")
    return answers_path


# --------------------- RAGAS evaluation ------------------- #

def to_single_turn_samples(
    answers: List[Dict[str, Any]],
    logger=None
) -> List[SingleTurnSample]:
    """
    Build RAGAS SingleTurnSample list with contexts.
    Expected per answer row: {"id", "question", "answer", "contexts": List[str]}
    Expected per question row: {"id", "answers_str": List[str]}
    """

    samples: List[SingleTurnSample] = []
    for a in answers:
        qid = a.get("id")
        question = a.get("question", "")
        pred = a.get("answer", "")
        reference = ", ".join(a.get("gold", [])) or ""


        if not qid or not question or not pred:
            if logger:
                logger.warning(f"Skipping malformed answer row (id={qid})")
            continue


        samples.append(
            SingleTurnSample(
                user_input=str(question),
                response=str(pred),
                reference=reference,
            )
        )

    return samples


def run_ragas_evaluation(
    answers_path: Path,
    output_dir: Path,
    ragas_batch_size: int = 8,
    judge_model: Optional[str] = None,
    embed_model: str = "nomic-embed-text",
    ollama_host: Optional[str] = None,
    logger=None
):
    """
    Evaluate with full RAGAS metrics, using Ollama LLM + embeddings.
    """
    answers = load_jsonl(answers_path)


    # smoke test
    # answers = answers[:2]  # limit to first 10 for quick testing

    samples = to_single_turn_samples(
        answers=answers,
        logger=logger
    )
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        # context_precision,
        # context_recall,
        # faithfulness,
        # answer_relevancy,
        answer_correctness,
    ]

    # Backends, Here we use LangChain wrapper for compatibility
    llm = OllamaLLM(
        model=os.getenv("LLM_MODEL_NAME") or "llama3.1",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434",
        temperature=0.0,
        format='json'
    )
    embeddings = OllamaEmbeddings(
        model=os.getenv("LLM_EMBEDDING_MODEL_NAME") or "nomic-embed-text",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434"
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,                  # required for faithfulness/answer_* metrics
        embeddings=embeddings,    # required for context_* and answer_relevancy
        experiment_name='lightrag_evaluation',
        show_progress=True,
        batch_size=ragas_batch_size,
    )

    result_df = results.to_pandas()

    metrics_path = os.path.join(output_dir, "evaluation.metrics")
    write_txt(metrics_path, str(results))

    df_path = os.path.join(output_dir, "evaluation.csv")
    write_df_to_csv(df_path, result_df)

    print(f"Evaluation results saved to {metrics_path} and {df_path}")


# ---------------------------- Main ---------------------------- #

def main() -> None:
    load_dotenv()
    args = add_cli_args()

    # Timestamped log
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"lightrag_pipeline_{ts}.log"
    log = configure_logger(log_path)
    log.info("Starting LightRAG pipeline (with context persistence)")

    # Ensure output dir exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read questions
    questions = load_jsonl(args.questions_path)
    log.info(f"Loaded {len(questions)} questions from {args.questions_path}")

    answers_path = args.output_dir / args.answers_filename

    # Answer generation (skip if file exists and flag not set)
    should_generate = args.generate_answers or (not answers_path.exists())
    if should_generate:
        log.info("Running answer generation with LightRAG")
        asyncio.run(
            generate_answers_lightrag(
                questions=questions,
                storage_dir=args.storage_dir,
                output_dir=args.output_dir,
                answers_filename=args.answers_filename,
                mode=args.mode,
                logger=log,
            )
        )
    else:
        log.info("Answers file exists and --generate_answers not set; skipping generation.")

    # RAGAS evaluation (full metrics, requires contexts)
    if args.evaluate:
        log.info("Running RAGAS evaluation (full metrics)")

        run_ragas_evaluation(
            answers_path=answers_path,
            output_dir=args.output_dir,
            ragas_batch_size=args.ragas_batch_size,
            judge_model=args.judge_model,
            embed_model=args.embed_model,
            ollama_host=args.ollama_host,
            logger=log
        )
    else:
        log.info("Evaluation step skipped (no --evaluate).")

    log.info("Pipeline finished.")


if __name__ == "__main__":
    main()
