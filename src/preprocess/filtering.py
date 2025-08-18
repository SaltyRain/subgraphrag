from tqdm import tqdm
from typing import List, Dict, cast

from src.utils.logger import logger
from src.types import AnnotatedQuestion, QuestionWithEntityTitles, TagMeAnnotation, SRTKStructure


def filter_annotated_questions(
        data: List[AnnotatedQuestion],
        link_probability: float = 0.004,
        rho: float = 0.004
) -> List[QuestionWithEntityTitles]:
    """
    Filter annotated questions based on link probability and rho.

    Args:
        data (List[AnnotatedQuestion]): List of question dicts with 'annotations' field.
        link_probability (float): Minimum link probability threshold.
        rho (float): Minimum rho threshold.

    Returns:
        List[Dict]: Filtered list of questions with valid annotations.
    """
    filtered_data = []

    for item in tqdm(data, desc="🔍 Filtering Annotations", unit="question"):
        annotations_raw = item["annotations"]
        annotations: List[TagMeAnnotation] = annotations_raw if isinstance(annotations_raw, list) else []

        # Filter annotations
        filtered_annotations = [
            ann for ann in annotations
            if isinstance(ann, dict) and
               ann.get("link_probability", 0.0) >= link_probability and
               ann.get("rho", 0.0) >= rho
        ]

        if not filtered_annotations:
            logger.warning(f"No valid annotations for question ID {item['id']}. Skipping.")
            continue

        titles = [ann["title"].strip().replace(" ", "_") for ann in filtered_annotations if "title" in ann and ann["title"].strip()]

        if titles:
            filtered_data.append(cast(QuestionWithEntityTitles, {
                "id": item["id"],
                "question": item["question"],
                "answer_entities": item["answer_entities"],
                "titles": titles
            }))

    logger.info(f"Filtered down to {len(filtered_data)} questions with valid annotation titles.")
    return filtered_data


def filter_answer_entities(data: List[SRTKStructure]) -> List[SRTKStructure]:
    return [
        item for item in data
        # If at least one answer and all starts with "Q" (Wikidata ID)
        if item["answer_entities"]
        and all(a.startswith("Q") for a in item["answer_entities"])
    ]
