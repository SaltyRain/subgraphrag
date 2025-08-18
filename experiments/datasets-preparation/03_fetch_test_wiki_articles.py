import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.preprocess.articles_fetching import fetch_articles
from src.types import SRTKStructure
from src.utils.fs import ensure_directory
from src.utils.logger import logger, configure_logger

from typing import List

# Configure logging level from .env
load_dotenv()
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def fetch_test_wiki_articles(
        input_path: Path,
        output_dir: Path,
        mapper_path: Path,
        intermediate_dir: Path = None,
):
    if not input_path.exists():
        logger.error(f"❌ Input path {input_path} does not exist.")
        return

    if not mapper_path.exists():
        logger.error(f"❌ Mapper dump path {mapper_path} does not exist.")
        return

    if intermediate_dir:
        ensure_directory(intermediate_dir, "intermediate")

    with open(input_path, "r", encoding="utf-8") as f:
        questions: List[SRTKStructure] = [json.loads(line) for line in f]


    entities = set()
    for question in questions:
        entities.update(question["question_entities"])

    logger.info(f"✅ Found {len(entities)} unique entities in the test set.")
    if intermediate_dir:
        unique_entities_path = intermediate_dir / "01_unique_entities.txt"
        with open(unique_entities_path, "w", encoding="utf-8") as f:
            for entity in sorted(entities):
                f.write(entity + "\n")
        logger.info(f"💾 Unique entities saved to {unique_entities_path}")

    # Fetch articles for the unique entities
    fetch_articles(
        entities=list(entities),
        mapper_path=mapper_path,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles based on question entities from the test webqsp-wd dataset.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the input test set JSONL file.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to save the fetched articles.")
    parser.add_argument("--mapper_path", type=Path, required=True, help="Path to the Wikidata ID mapper dump.")
    parser.add_argument("--intermediate_dir", type=Path, default=None, help="Directory for intermediate files.")
    parser.add_argument("--log_path", type=Path,  default=Path("logs/preprocess/fetch_wikipedia_articles.log"), help="Path to save the log file.")

    args = parser.parse_args()
    logger = configure_logger(args.log_path)

    fetch_test_wiki_articles(
        input_path=args.input_path,
        output_dir=args.output_dir,
        mapper_path=args.mapper_path,
        intermediate_dir=args.intermediate_dir,
    )