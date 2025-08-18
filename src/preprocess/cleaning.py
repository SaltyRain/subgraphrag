from typing import List

from src.utils.logger import logger
from src.types import RawTrainQuestion, CleanedQuestion

def clean_and_format_questions(data: List[RawTrainQuestion]) -> List[CleanedQuestion]:
    """
    Normalize and clean a list of question dicts by:
    - renaming fields to standard format,
    - removing None answers,
    - removing duplicates in answer_entities.

    Args:
        data (List[Dict]): Raw list of question dicts (e.g. from WebQSP).

    Returns:
        List[Dict]: Cleaned and standardized list of questions.
    """
    cleaned_data = []
    for item in data:
        question_id = item.get("questionid", "")
        question_text = item.get("utterance", "")
        answer_entities = item.get("answers", [])

        # Clean answer entities: remove None and duplicates
        cleaned_answers = list({a for a in answer_entities if a is not None})

        cleaned_data.append({
            "id": question_id,
            "question": question_text,
            "answer_entities": cleaned_answers
        })

    logger.info(f"🧹 Cleaned {len(cleaned_data)} questions.")
    return cleaned_data

