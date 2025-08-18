from typing import List, cast
from tqdm import tqdm

from src.utils.logger import logger
from src.types import CleanedQuestion, AnnotatedQuestion
from src.utils.tagme_annotate import tagme_annotate


def link_entities_to_questions(data: List[CleanedQuestion]) -> List[AnnotatedQuestion]:
    """
    Link entities to questions using the TagMe API.

    Args:
        data (List[CleanedQuestion]): List of question dicts with 'question' field.

    Returns:
        List[AnnotatedQuestion]: List of questions with linked entities.
    """
    linked_data = []
    for item in tqdm(data, desc="🔗 Entity Linking with TagMe", unit="question"):
        question = item["question"]

        if (not question) or (not isinstance(question, str)):
            logger.warning(f"Invalid question text: {question}")
            continue

        annotations = tagme_annotate(text=question)

        if not annotations:
            logger.warning(f"No annotations found for question: {question}")
            annotations = []

        linked_data.append(cast(AnnotatedQuestion, {
            "id": item["id"],
            "question": question,
            "answer_entities": item.get("answer_entities", []),
            "annotations": annotations
        }))

    return linked_data
