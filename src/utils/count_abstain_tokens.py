import json
import argparse
from pathlib import Path
import re
import unicodedata


from src.prompts.answer_prompt import ABSTAIN_TOKEN


PUNCT_ONLY_RE = re.compile(r'^[\s\.\!\?;,:\-–—]*$')  # regex for "empty" punctuation

def is_strict_abstain(raw: str, token: str = ABSTAIN_TOKEN) -> bool:
    """
    Return True only if the answer equals the abstain token (e.g., NO_ANSWER),
    optionally wrapped in trivial shells (quotes/backticks/parentheses/code fences)
    and/or surrounded by whitespace or pure punctuation. Any extra words -> False.
    """
    if raw is None:
        return False

    # Normalize Unicode (NFKC) and strip whitespace
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if not s:
        return False

    # Handle code fences: ```NO_ANSWER``` or ```\nNO_ANSWER\n```
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()

    # Handle single wrappers: "NO_ANSWER", 'NO_ANSWER', `NO_ANSWER`
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("'") and s.endswith("'")) or \
       (s.startswith("`") and s.endswith("`")):
        s = s[1:-1].strip()

    # Handle a single pair of parentheses: (NO_ANSWER)
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        # Remove leading/trailing punctuation inside the parentheses
        while inner and PUNCT_ONLY_RE.match(inner[:1]):
            inner = inner[1:]
        while inner and PUNCT_ONLY_RE.match(inner[-1:]):
            inner = inner[:-1]
        s = inner.strip()

    # Allow trivial punctuation around the token:
    # e.g., "NO_ANSWER", "NO_ANSWER.", " NO_ANSWER ! "
    left = 0
    while left < len(s) and PUNCT_ONLY_RE.match(s[left:left+1]):
        left += 1
    right = len(s)
    while right > 0 and PUNCT_ONLY_RE.match(s[right-1:right]):
        right -= 1

    core = s[left:right].strip()

    return core == token

def count_no_answers(file_path: Path, abstain_token: str = ABSTAIN_TOKEN):
    """
    Read a JSONL answers file and count records where the answer is exactly the abstain token.

    Rules:
    - Count ONLY lines where `answer.strip()` == abstain_token.
    - Ignore lines that contain the token alongside other text (treated as noise).
    - Skip empty or malformed lines gracefully.
    """
    total = 0
    no_answer_count = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                # Skip empty lines
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupted/invalid JSON lines
                continue

            total += 1
            answer = str(record.get("answer", "")).strip()

            # Count only strict matches to the abstain token
            # if answer == abstain_token:
            #     no_answer_count += 1
            if is_strict_abstain(answer, abstain_token):
                no_answer_count += 1

    return total, no_answer_count


def main():
    parser = argparse.ArgumentParser(
        description="Count strict NO_ANSWER responses in a JSONL results file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the answers JSONL file"
    )
    parser.add_argument(
        "--token",
        default=ABSTAIN_TOKEN,
        help=f"Abstain token to match exactly (default: {ABSTAIN_TOKEN})"
    )
    args = parser.parse_args()


    # file_path = Path('../../results/exp2_baselines/lightrag/hybrid/answers.jsonl')

    total, no_answer_count = count_no_answers(args.file, args.token) #  file_path

    print(f"📊 Total records: {total}")
    print(f"🚫 NO_ANSWER records: {no_answer_count}")
    print(f"✅ With answers: {total - no_answer_count}")

    if total > 0:
        rate = 100.0 * no_answer_count / total
        print(f"📈 NO_ANSWER rate: {rate:.2f}%")

if __name__ == "__main__":
    main()
