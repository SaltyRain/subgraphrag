import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Optional

from dotenv import load_dotenv
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)


# RAGAS imports
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    # context_precision,
    # context_recall,
    # faithfulness,
    answer_correctness,
)

# LangChain-Ollama bindings for RAGAS LLM/Embeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings


from src.prompts.answer_prompt import SYSTEM_ANSWER, ABSTAIN_TOKEN
from src.utils.fs import write_df_to_csv, write_txt
# Custom logger
from src.utils.logger import configure_logger
from llama_index.core.prompts import PromptTemplate

load_dotenv()

# Prompt construction. Copies from `src/prompts/answer_prompt.py`
ANSWER_QA_TMPL = PromptTemplate(
    f"""{SYSTEM_ANSWER}

Question:
{{query_str}}

Context:
{{context_str}}

Output format:
- If exactly one answer: a single span (e.g., Nelson Mandela)
- If multiple answers: comma-separated list with a single space after each comma (e.g., A, B, C)
- If unknown/insufficient: {ABSTAIN_TOKEN}

Return ONLY the answer text."""
)




# ---------------------------
# Index building
# ---------------------------
def build_vector_index(
    documents_dir: Path,
    storage_dir: Path,
    llm: Ollama,
    embed_model: OllamaEmbedding,
    logger,
) -> VectorStoreIndex:
    """Build a vector index from the given the documents directory."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    documents = SimpleDirectoryReader(documents_dir).load_data(show_progress=True)
    logger.info(f"Loaded {len(documents)} documents from {documents_dir}")

    index = VectorStoreIndex.from_documents(
        documents=documents,
        llm=llm,
        embed_model=embed_model,
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=storage_dir)
    logger.info(f"Vector index built and saved to {storage_dir}")
    return index


def load_or_build_index(
        logger,
        rebuild=False,
        storage_dir: Path = Path("../../storage/naiverag"),
        documents_dir: Path = Path("../../outputs/processed/articles"),
        llm: Ollama = None,
        embed_model: OllamaEmbedding = None,

) -> BaseIndexStore or VectorStoreIndex:
    """Load index from storage or rebuild if needed."""
    need_rebuild = rebuild or (not storage_dir.exists())
    if need_rebuild:
        logger.info("🔧 Rebuilding vector index...")
        return build_vector_index(documents_dir, storage_dir, llm, embed_model, logger)
    else:
        logger.info(f"📦 Loading vector index from {storage_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
        return load_index_from_storage(storage_context, llm=llm, embed_model=embed_model)


# ---------------------------
# Answer generation
# ---------------------------
def generate_answers(
    llm: Ollama,
    embed_model: OllamaEmbedding,
    index: VectorStoreIndex,
    questions: List[Dict[str, Any]],
    answers_path: Path,
    similarity_top_k: int,
    logger,
):
    """Generate answers for given questions using the provided index."""
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    # Smoke test first 3 questions
    # questions = questions[:3]  # Uncomment for quick testing, remove in production

    # Load already processed question IDs
    processed_ids: Set[str] = set()
    if answers_path.exists():
        with answers_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    processed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    unanswered = [q for q in questions if q.get("id") not in processed_ids]
    logger.info(f"🧠 Questions to answer: {len(unanswered)} / {len(questions)}")

    if not unanswered:
        logger.info("✅ All questions already answered. Skipping generation.")
        return

    query_engine = index.as_query_engine(
        llm=llm,
        embed_model=embed_model,
        similarity_top_k=similarity_top_k,
        text_qa_template=ANSWER_QA_TMPL,
        response_mode="compact",
    )

    def serialize_node(hit) -> Dict[str, Any]:
        return {
            "id": str(hit.node.node_id),
            "score": float(hit.score) if hit.score is not None else None,
            "text": hit.node.text,
            "metadata": hit.node.metadata,
        }

    with answers_path.open("a", encoding="utf-8") as f, tqdm(total=len(unanswered), desc="🧠 Generating") as pbar:
        for q in unanswered:
            qid = q.get("id")
            try:
                resp = query_engine.query(q["question"])
                result = {
                    "id": qid,
                    "question": q["question"],
                    "answer": resp.response if hasattr(resp, "response") else str(resp),
                    "gold": q.get("answers_str", []),
                    "source_nodes": [serialize_node(n) for n in getattr(resp, "source_nodes", [])],
                }
                logger.info(f"[{qid}] {result['answer'][:120]}...")
            except Exception as e:
                logger.exception(f"Error answering id={qid}: {e}")
                result = {"id": qid, "question": q.get("question"), "answer": f"ERROR: {e}", "source_nodes": []}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)

    logger.info(f"✅ Answers written to {answers_path}")


# ---------------------------
# RAGAS utils
# ---------------------------
def to_single_turn_samples(rows: Iterable[Dict[str, Any]]) -> List[SingleTurnSample]:
    """Convert rows with answers into SingleTurnSample list for RAGAS."""
    samples: List[SingleTurnSample] = []
    for r in rows:
        question = r.get("question", "")
        pred = r.get("answer", "") or ""
        golds = r.get("answers_str", []) or []
        ctx = [n.get("text", "").strip() for n in r.get("source_nodes", []) if n.get("text", "").strip()]

        if not question or not pred or not ctx:
            continue

        reference = ", ".join(golds) if golds else ""
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=ctx,
                response=str(pred),
                reference=reference,
            )
        )
    return samples


def _contexts_from_source_nodes(row: Dict[str, Any], top_k: Optional[int] = None) -> List[str]:
    nodes = row.get("source_nodes") or []
    # если есть score — отсортируем по убыванию
    try:
        nodes = sorted(nodes, key=lambda n: float(n.get("score", 0.0)), reverse=True)
    except Exception:
        pass
    if isinstance(top_k, int) and top_k > 0:
        nodes = nodes[:top_k]
    ctx = []
    for n in nodes:
        t = (n.get("text") or "").strip()
        if t:
            ctx.append(t)
    return ctx

def _to_single_turn_samples(
    answered_rows: Iterable[Dict[str, Any]],
    # top_k_ctx: Optional[int] = None,
    # drop_if_no_gold: bool = False,
) -> List[SingleTurnSample]:
    samples: List[SingleTurnSample] = []
    for r in answered_rows:
        question = r.get("question") or ""
        pred = r.get("answer") or ""
        reference = ", ".join(r.get("gold", [])) or ""


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
    results_dir: Path,
    batch_size: int,
    logger,
    # top_k_ctx: Optional[int] = None,
):
    """Run RAGAS evaluation on generated answers."""

    # Backends, Here we use LangChain wrapper for compatibility
    llm = OllamaLLM(
        model=os.getenv("LLM_MODEL_NAME") or "llama3.1",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434",
        temperature=0.0,
    )
    embeddings = OllamaEmbeddings(
        model=os.getenv("LLM_EMBEDDING_MODEL_NAME") or "nomic-embed-text",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434"
    )


    if not answers_path.exists():
        logger.error(f"❌ Answers file not found: {answers_path}")
        return

    with answers_path.open("r", encoding="utf-8") as f:
        answered = [json.loads(line) for line in f if line.strip()]

    # smoke test first 3 answers
    # answered = answered[:3]  # Uncomment for quick testing, remove in production

    samples = _to_single_turn_samples(
        answered_rows=answered,
    )

    if not samples:
        logger.warning("⚠️ No valid samples for RAGAS.")
        return

    metrics = [answer_correctness]

    logger.info(f"🧪 RAGAS evaluating {len(samples)} samples...")

    dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        experiment_name='naiverag_evaluation',
        show_progress=True,
        batch_size=max(8, batch_size),
    )

    #  Convert the result to a pandas DataFrame
    result_df = results.to_pandas()

    metrics_path = os.path.join(results_dir, "evaluation.metrics")
    write_txt(metrics_path, str(results))

    df_path = os.path.join(results_dir, "evaluation.csv")
    write_df_to_csv(df_path, result_df)

    print(f"Evaluation results saved to {metrics_path} and {df_path}")



# ---------------------------
# Argument parser
# ---------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NaiveRAG baseline: index -> answer -> evaluate (RAGAS).")

    # Paths and data
    p.add_argument("--documents_dir", type=Path, default=Path("../../outputs/processed/articles"),
                   help="Directory containing input documents.")
    p.add_argument("--storage_dir", type=Path, default=Path("../../storage/naiverag"),
                   help="Directory for persisted vector index.")
    p.add_argument("--questions_path", type=Path, default=Path("../../outputs/processed/test.jsonl"),
                   help="JSONL file with questions (fields: id, question, answers_str).")
    p.add_argument("--results_dir", type=Path, default=Path("../../outputs/naiverag/results"),
                   help="Directory to store answers and evaluation artifacts.")
    p.add_argument("--answers_filename", type=str, default="answers.jsonl",
                   help="Filename for generated answers inside results_dir.")

    # Stage control flags
    p.add_argument("--rebuild_index", action="store_true",
                   help="Force rebuild of index (otherwise loads from storage_dir if exists).")
    p.add_argument("--generate_answers", action="store_true",
                   help="Force regeneration of answers even if answers file exists.")
    p.add_argument("--evaluate", action="store_true",
                   help="Run RAGAS evaluation after answers are ready.")

    # Retrieval params
    p.add_argument("--similarity_top_k", type=int, default=2, help="Top-k chunks to retrieve per query.")

    # Logging
    p.add_argument("--log_dir", type=Path, default=Path("../../logs/naiverag"),
                   help="Directory for timestamped logs.")

    # RAGAS params
    p.add_argument("--ragas_batch_size", type=int, default=4, help="Batch size for RAGAS evaluation.")

    return p

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log_dir / f"naiverag_{timestamp}.log"
    logger = configure_logger(log_path)
    logger.info(f"▶️ Starting pipeline. Results dir: {args.results_dir}")

    # LlamaIndex wrappers
    llm = Ollama(
        model=os.getenv("LLM_MODEL_NAME") or "llama3.1",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434",
    )
    embed_model = OllamaEmbedding(
        model_name=os.getenv("LLM_EMBEDDING_MODEL_NAME") or "nomic-embed-text",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://localhost:11434",
    )

    index = load_or_build_index(
        rebuild=args.rebuild_index,
        storage_dir=args.storage_dir,
        documents_dir=args.documents_dir,
        llm=llm,
        embed_model=embed_model,
        logger=logger,
    )

    with args.questions_path.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(questions)} questions from {args.questions_path}")

    answers_path = args.results_dir / args.answers_filename



    if args.generate_answers or not answers_path.exists():
        logger.info("🧩 Generating answers…")
        generate_answers(
            llm=llm,
            embed_model=embed_model,
            index=index,
            questions=questions,
            answers_path=answers_path,
            similarity_top_k=args.similarity_top_k,
            logger=logger,
        )
    else:
        logger.info(f"✅ Answers already exist at {answers_path}. Skipping generation.")

    if args.evaluate:
        run_ragas_evaluation(
            answers_path=answers_path,
            results_dir=args.results_dir,
            batch_size=args.ragas_batch_size,
            logger=logger,
        )
    else:
        logger.info("ℹ️ Evaluation stage skipped (use --evaluate).")

    logger.info("🏁 Done.")


if __name__ == "__main__":
    main()
