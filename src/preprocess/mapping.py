import re

from pathlib import Path
from typing import List, cast
from tqdm import tqdm
from wikimapper import WikiMapper

from src.utils.logger import logger
from src.types import QuestionWithEntityTitles, SRTKStructure

def map_titles_to_wikidata_ids(
    data: List[QuestionWithEntityTitles],
    mapper_path: Path
) -> List[SRTKStructure]:
    """
    Map Wiki page titles to Wikidata IDs using a mapper dump.

    Args:
        data (List[QuestionWithEntityTitles]): List of questions with entity titles.
        mapper_path (Path): Path to the mapper dump for Wikidata IDs.

    Returns:
        List[SRTKStructure]: List of questions with mapped Wikidata IDs.
    """

    if not mapper_path.exists():
        logger.error(f"Mapper path does not exist: {mapper_path}")
        return []

    mapper = WikiMapper(str(mapper_path))
    result = []

    for item in tqdm(data, desc="🗺️ Mapping Titles to Wikidata IDs", unit="question"):
        titles = item["titles"]
        mapped_entities = []

        for title in titles:
            title_clean = re.sub(r"\s+", "_", title.strip())
            if title_clean:
                try:
                    wikidata_id = mapper.title_to_id(title_clean)
                    if wikidata_id:
                        mapped_entities.append(wikidata_id)
                    else:
                        logger.warning(f"No Wikidata ID found for title: {title_clean}")
                except Exception as e:
                    logger.error(f"Error mapping title '{title_clean}': {e}")

        result.append(cast(SRTKStructure, {
            "id": item["id"],
            "question": item["question"],
            "question_entities": mapped_entities,
            "answer_entities": item["answer_entities"]
        }))

    return result
