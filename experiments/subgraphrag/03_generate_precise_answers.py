import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import tqdm

from src.llm.ollama import agenerate_llm_response
from src.prompts.answer_prompt import SYSTEM_ANSWER, ABSTAIN_TOKEN, ANSWER_PROMPT
from src.utils.logger import configure_logger




def _postprocess_answer(text: str) -> str:
    """
    Make the output stricter without changing substance:
    - strip whitespace
    - drop surrounding quotes/brackets
    - compress internal whitespace
    - normalize comma+space in lists
    - uppercase ABSTAIN token if it appears in any case
    """
    if not text:
        return ""
    t = text.strip()

    # Remove leading/trailing quotes or brackets
    t = re.sub(r'^[\s"“”\'\[\(]+', "", t)
    t = re.sub(r'[\s"“”\'\]\)]+$', "", t)

    # If model leaked the abstain token with case variation or extra dots
    if re.fullmatch(r'\s*no[_\s-]?answer\.?\s*', t, flags=re.IGNORECASE):
        return ABSTAIN_TOKEN

    # Replace newlines with spaces
    t = re.sub(r'\s+', " ", t)

    # Normalize list separators: ensure ", " and remove trailing comma
    t = re.sub(r'\s*,\s*', ", ", t).strip().rstrip(",")

    # Avoid trailing period if it looks like a list/item (not a sentence)
    if t.endswith(".") and ("," in t or len(t.split()) <= 6):
        t = t[:-1]

    return t


async def _gen_answer_for_question(
    record: Dict[str, Any],
    *,
    llm_timeout: int,
    logger,
) -> Optional[Dict[str, Any]]:
    """
    For one fused-context record, ask the LLM for the final answer.
    Expects fields:
      - id
      - question
      - fused_context
      - (optional) answers_str for reference
    Returns a result dict or None on failure.
    """
    qid = record.get("id")
    qtext = (record.get("question") or "").strip()
    fused_context = (record.get("fused_context") or "").strip()
    answers_gold = record.get("answers_str", None)

    if not fused_context:
        logger.warning(f"Question {qid}: empty fused_context.")
        return None

    prompt = ANSWER_PROMPT.format(
        question=qtext,
        context=fused_context,
        abstain=ABSTAIN_TOKEN,
    )

    try:
        raw = await agenerate_llm_response(
            prompt=prompt,
            system_prompt=SYSTEM_ANSWER,
            stream=False,
            timeout=llm_timeout,
        )
        ans = _postprocess_answer((raw or "").strip())

        if not ans:
            logger.warning(f"Question {qid}: empty answer from LLM.")
            return None

        result: Dict[str, Any] = {
            "id": qid,
            "question": qtext,
            "fused_context": fused_context,
            "answer": ans,
        }
        if answers_gold is not None:
            result["gold"] = answers_gold
        return result

    except Exception as e:
        logger.warning(f"Question {qid}: failed to generate answer: {e}")
        return None


async def generate_answers_async(
    fused_records: List[Dict[str, Any]],
    *,
    output_path: str,
    logger,
    llm_timeout: int = 60,
) -> None:
    """
    Sequentially process fused-context records; for each, call the LLM ONCE to produce the final answer.
    Append each result immediately to output_path as JSONL.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Truncate output file once
    with open(output_path, "w", encoding="utf-8") as _:
        pass

    for rec in tqdm(fused_records, desc="Generating answers"):
        out = await _gen_answer_for_question(
            rec,
            llm_timeout=llm_timeout,
            logger=logger,
        )
        if out is None:
            continue
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


async def async_main():
    parser = argparse.ArgumentParser(
        description="Generate final answers from fused, query-focused contexts."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to JSONL produced by fuse_context (with 'fused_context').")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write answers (JSONL).")
    parser.add_argument("--log_dir", type=str, default="logs/subgraphrag/generate_answers",
                        help="Directory to save logs.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="LLM request timeout in seconds.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(f"{args.log_dir}/generate_answers_{timestamp}.log")
    logger = configure_logger(log_path)

    # Read fused-context JSONL
    with open(args.input_path, "r", encoding="utf-8") as f:
        fused_records = [json.loads(line) for line in f]

    # # For a quick smoke test:
    # fused_records = fused_records[:3]

    await generate_answers_async(
        fused_records=fused_records,
        output_path=args.output_path,
        logger=logger,
        llm_timeout=args.timeout,
    )

    logger.info(f"Answers written to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
