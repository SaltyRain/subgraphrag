import argparse
import os
import json

from pathlib import Path
from typing import List
from dotenv import load_dotenv

from src.preprocess.cleaning import clean_and_format_questions
from src.preprocess.entity_linking import link_entities_to_questions
from src.preprocess.filtering import filter_annotated_questions, filter_answer_entities
from src.preprocess.mapping import map_titles_to_wikidata_ids
from src.types import RawTrainQuestion
from src.utils.fs import save_intermediate, ensure_directory
from src.utils.logger import logger, configure_logger

# Configure logging level from .env
load_dotenv()
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

def prepare_train_dataset(
        input_path: Path,
        output_path: Path,
        mapper_path: Path,
        intermediate_dir: Path = None,
        # Entity Linking Parameters
        rho: float = 0.004,
        link_probability: float = 0.004,
):
    """
    Prepares the training dataset by reading the input JSON file and writing it to the output path.

    Args:
        input_path (Path): Path to the input JSON file.
        output_path (Path): Path where the output JSON file will be saved.
        mapper_path (Path): Path to the mapper dump for Wikidata IDs.
        intermediate_dir (Path, optional): Directory for intermediate files. Defaults to None.
        rho (float): Threshold for rho in entity linking. Defaults to 0.004.
        link_probability (float): Threshold for link probability in entity linking. Defaults to 0.004.
    """
    if not input_path.exists():
        logger.error(f"❌ Input path {input_path} does not exist.")
        return

    if not mapper_path.exists():
        logger.error(f"❌ Mapper dump path {mapper_path} does not exist.")
        return

    if intermediate_dir:
        ensure_directory(intermediate_dir, "intermediate")

    with open(input_path, 'r', encoding='utf-8') as infile:
        data: List[RawTrainQuestion] = json.load(infile)


    # 1. Clean and format the questions
    cleaned_data = clean_and_format_questions(data)
    if intermediate_dir:
        save_intermediate(cleaned_data, intermediate_dir / "01_cleaned_questions.json", "💾 Cleaned data")

    # 2. Entity linking
    linked_data = link_entities_to_questions(cleaned_data)
    if intermediate_dir:
        save_intermediate(linked_data, intermediate_dir / "02_linked_questions.json", "🔗 Linked data")

    # 3. Filtering annotated questions based on rho and link_probability
    filtered_data = filter_annotated_questions(linked_data, link_probability=link_probability, rho=rho)
    if intermediate_dir:
        save_intermediate(filtered_data, intermediate_dir / "03_filtered_questions.json", "🔍 Filtered data")

    # 4. Map Wiki page titles to Wikidata IDs
    mapped_data = map_titles_to_wikidata_ids(filtered_data, mapper_path)
    if intermediate_dir:
        save_intermediate(mapped_data, intermediate_dir / "04_mapped_questions.json", "🗺️ Mapped data")

    # NOTE: The resulting dataset is noisy - some questions have dates instead on wikidata IDs
    # others are missing answer_entities because in the original dataset there were NULLs in ids,
    # but answer_str (names) were there. We will filter them out.
    # 5. Filter out noisy questions and get final structure for SRTK
    final_data = filter_answer_entities(mapped_data)

    # Ensure output directory exists
    ensure_directory(output_path.parent, "output")

    # Save final dataset in JSONL format
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in final_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

    # Stats
    logger.info(f"✅ Final JSONL dataset saved to {output_path} with {len(final_data)} questions.")
    logger.info("📊 Original dataset size: %d", len(data))
    logger.info("📉 Final dataset size: %d", len(final_data))
    logger.info("📈 Retained %.2f%% of data", (len(final_data) / len(data)) * 100 if data else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset for SRTK training retriever.")

    parser.add_argument("--input_path", type=Path, required=True, help="Path to raw train JSON file")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the prepared dataset")
    parser.add_argument("--mapper_path", type=Path, required=True, help="Path to the WikiMapper SQL dump")
    parser.add_argument("--intermediate_dir", type=Path, required=False, help="Directory to save intermediate files")
    parser.add_argument("--rho", type=float, default=0.004, help="Entity linking rho threshold")
    parser.add_argument("--link_probability", type=float, default=0.004, help="Entity linking probability threshold")
    parser.add_argument("--log_path", type=Path, default=Path("logs/preprocess/prepare_train_dataset.log"),
                        help="Path to save the log file")

    args = parser.parse_args()
    logger = configure_logger(args.log_path)

    prepare_train_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        mapper_path=args.mapper_path,
        intermediate_dir=args.intermediate_dir,
        rho=args.rho,
        link_probability=args.link_probability,
    )
