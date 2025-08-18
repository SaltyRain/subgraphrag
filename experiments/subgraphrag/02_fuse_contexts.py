import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from datasets import tqdm

from src.llm.ollama import agenerate_llm_response
from src.utils.logger import configure_logger



SYSTEM_FUSION = (
    "You are a careful data-to-text assistant. "
    "You merge compact knowledge snippets into a single, concise, factual context. "
    "Preserve only information that directly helps answer the question. "
    "Never invent facts. Deduplicate aggressively. Be terse and precise."
)

CONTEXT_FUSION_PROMPT = """You are given a question and several compact, head-centric snippets derived from Wikidata 1–2 hop paths.
Merge them into ONE concise, self-contained context that maximizes relevance to the question.

Guidelines:
- Keep only facts that are directly helpful for answering the question.
- Deduplicate overlapping content across snippets.
- Prefer exact entity/relation wording; avoid speculation.
- Be brief: target ≤ {max_words} words.
- Output ONLY the final context text (no preface, no bullets, no headings).

Question:
{question}

Snippets:
{snippets}
"""


def _format_snippets(contexts_map: Dict[str, str], max_heads: Optional[int], sep: str = "\n\n") -> Tuple[str, List[str]]:
    """
    Turn {"HEAD": "compact text", ...} into a single string that the prompt can consume.
    We preserve insertion order (Python 3.7+) so it’s deterministic w.r.t. previous stage.
    Returns the snippets block and the ordered list of heads actually used.
    """
    items = list(contexts_map.items())
    if max_heads is not None:
        items = items[:max_heads]
    used_heads: List[str] = []
    blocks: List[str] = []
    for head, text in items:
        head = head or ""
        used_heads.append(head)
        # Simple, readable frame per snippet
        blocks.append(f"=== {head} ===\n{text}".strip())
    return sep.join(blocks), used_heads


async def _gen_fused_context_for_question(
    question: Dict[str, Any],
    *,
    max_heads: Optional[int],
    max_words: int,
    llm_timeout: int,
    logger,
) -> Optional[Dict[str, Any]]:
    """
    Build a single fused context for one question by prompting the LLM with all compact snippets.
    Returns a record to append to the output file, or None on failure.
    """
    qid = question.get("id")
    qtext = question.get("question", "")
    answers = question.get("answers_str", None)
    contexts_map: Dict[str, str] = question.get("contexts") or {}

    if not contexts_map:
        logger.warning(f"Question {qid}: no compact contexts found.")
        return None

    snippets_block, used_heads = _format_snippets(contexts_map, max_heads=max_heads)
    prompt = CONTEXT_FUSION_PROMPT.format(
        question=qtext.strip(),
        snippets=snippets_block,
        max_words=max_words,
    )

    try:
        fused = await agenerate_llm_response(
            prompt=prompt,
            system_prompt=SYSTEM_FUSION,
            stream=False,
            timeout=llm_timeout,
        )
        fused = (fused or "").strip()
        if not fused:
            logger.warning(f"Question {qid}: empty fused context from LLM.")
            return None

        record: Dict[str, Any] = {
            "id": qid,
            "question": qtext,
            "fused_context": fused,
            "used_heads": used_heads,  # transparency of what was fed
        }
        if answers is not None:
            record["answers_str"] = answers
        return record

    except Exception as e:
        logger.warning(f"Question {qid}: failed to fuse context: {e}")
        return None


async def generate_fused_contexts_async(
    questions: List[Dict[str, Any]],
    *,
    output_path: str,
    logger,
    max_heads: Optional[int] = None,   # None = use all heads present
    max_words: int = 120,              # soft target for the LLM
    llm_timeout: int = 60,
) -> None:
    """
    Sequentially process questions; for each, call the LLM ONCE to produce a single fused context.
    Append each result immediately to output_path as JSONL.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Truncate output file once
    with open(output_path, "w", encoding="utf-8") as _:
        pass

    for q in tqdm(questions, desc="Fusing contexts"):
        rec = await _gen_fused_context_for_question(
            q,
            max_heads=max_heads,
            max_words=max_words,
            llm_timeout=llm_timeout,
            logger=logger,
        )
        if rec is None:
            continue
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def async_main():
    parser = argparse.ArgumentParser(
        description="Fuse compact contexts into a single prompt context per question (no head post-ranking)."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to JSONL produced by build_compact_contexts (per-question 'contexts' map).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write fused contexts (JSONL).")
    parser.add_argument("--log_dir", type=str, default="logs/subgraphrag/fuse_contexts",
                        help="Directory to save logs.")
    parser.add_argument("--max_heads", type=int, default=None,
                        help="Max number of heads (snippets) to include per question. None = use all.")
    parser.add_argument("--max_words", type=int, default=120,
                        help="Target word budget for the fused context.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="LLM request timeout in seconds.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(f"{args.log_dir}/fuse_contexts_{timestamp}.log")
    logger = configure_logger(log_path)

    # Read compact-context JSONL
    with open(args.input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # #test first 3  questions
    # questions = questions[:3]  # For testing, remove in production

    await generate_fused_contexts_async(
        questions=questions,
        output_path=args.output_path,
        logger=logger,
        max_heads=args.max_heads,
        max_words=args.max_words,
        llm_timeout=args.timeout,
    )

    logger.info(f"Fused contexts written to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
