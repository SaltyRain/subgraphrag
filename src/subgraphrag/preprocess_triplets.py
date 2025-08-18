from typing import List, Set, Dict
from src.types import SRTKStructureWithTriplets
from src.utils.logger import logger

def collect_unique_qids_from_triplets(data: List[SRTKStructureWithTriplets]) -> Set[str]:
    qids: Set[str] = set()

    for item in data:
        triplets = item.get("triplets", [])
        for triplet in triplets:
            if len(triplet) == 3:
                subj, pred, obk = triplet
                qids.update([subj, pred, obk])
            else:
                logger.warning(f"⚠️ Skipping invalid triplet: {triplet}")

    logger.info(f"Collected {len(qids)} unique QIDs from triplets.")
    return qids


def _triplet_to_text_from_metadata(
    subj_id: str,
    pred_id: str,
    obj_id: str,
    labels_map: Dict[str, str]
) -> str:
    def format_label(qid: str) -> str:
        return labels_map.get(qid, "unknown")

    subj = format_label(subj_id)
    pred = format_label(pred_id)
    obj = format_label(obj_id)

    return f"{subj} — {pred} — {obj}"



def convert_triplets_to_texts_for_question(
    question_item: SRTKStructureWithTriplets,
    labels_map: Dict[str, str]
) -> List[str]:
    """
    Converts the triplets of a question into textual format using enriched metadata.
    """
    result = []
    for triplet in question_item["triplets"]:
        subj, pred, obj = triplet
        try:
            triplet_text = _triplet_to_text_from_metadata(subj, pred, obj, labels_map)
            result.append(triplet_text)
        except Exception as e:
            logger.info(f"❌ Failed to convert triplet {triplet}: {e}")
    return result