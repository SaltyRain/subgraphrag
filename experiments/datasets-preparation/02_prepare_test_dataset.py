import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from src.utils.logger import configure_logger, logger

# Load .env and set verbosity
load_dotenv()
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def prepare_test_dataset(input_path: Path, output_path: Path):
    """
    Converts the WebQSP-WD test dataset to the format required for evaluation.

    Output format:
    {
        "id": "WebQTest-12",
        "question": "who is governor of ohio 2011?",
        "question_entities": ["Q1397"],
        "answer_entities": ["Q69319", "Q744819", "Q465295"]
        "answers_str": ["edwin l. porter", "edwin l. porter", "edwin l. porter"]
    }
    """
    if not input_path.exists():
        logger.error(f"❌ Input path {input_path} does not exist.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)


    skipped = 0
    written = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            qid = item.get("questionid")
            question = item.get("utterance")
            answer_entities = item.get("answers", [])
            answers_str = item.get("answers_str", [])

            # remove duplicates
            answer_entities = list(dict.fromkeys(answer_entities))
            answers_str = list(dict.fromkeys(answers_str))

            # skip if no answers
            if not answer_entities:
                skipped += 1
                continue


            entities = item.get("entities", [])
            if entities and isinstance(entities[0], dict):
                linkings = entities[0].get("linkings", [])
                question_entities = [l[0] for l in linkings if l]
            else:
                question_entities = []

            converted = {
                "id": qid,
                "question": question,
                "question_entities": question_entities,
                "answer_entities": answer_entities,
                "answers_str": answers_str
            }

            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"✅ Test dataset saved to {output_path} with {written} questions. Skipped {skipped} empty items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test dataset for evaluation.")

    parser.add_argument("--input_path", type=Path, required=True, help="Path to WebQSP-WD test JSON file")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the formatted test dataset")
    parser.add_argument("--log_path", type=Path, default=Path("logs/preprocess/prepare_test_dataset.log"),
                        help="Path to save the log file")

    args = parser.parse_args()

    configure_logger(args.log_path)

    prepare_test_dataset(
        input_path=args.input_path,
        output_path=args.output_path
    )
