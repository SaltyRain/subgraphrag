import argparse
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Dict, Set, List
import json

from tqdm import tqdm

from src.sparql.utils import get_entities_labels_batch
from src.subgraphrag.preprocess_triplets import collect_unique_qids_from_triplets
from src.types import SRTKStructureWithTriplets
from src.utils.logger import configure_logger

CHECKPOINT_INTERVAL = 100
BATCH_SIZE = 20

def main(
    labels_map_path: Path,
    input_path: Path,
    intermediate_dir: Path = None,
):

    if not input_path.exists():
        logger.info(f"❌ Input path {input_path} does not provided.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        input_data: List[SRTKStructureWithTriplets] = [json.loads(line) for line in f]

    # 1. Collect unique QIDs from triplets
    unique_qids = collect_unique_qids_from_triplets(input_data)
    logger.info(f"✅ Found {len(unique_qids)} unique QIDs.")

    labels_map: Dict[str, str] = {}
    if labels_map_path and labels_map_path.exists():
        logger.info(f"📁 Loading labels map from  {labels_map_path}")
        with open(labels_map_path, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)
    else:
        logger.info(f"📁 No labels found at {labels_map_path}, will create new.")

    # 3. Determine missing or invalid QIDs
    missing_qids: Set[str] = set()

    for qid in unique_qids:
        label = labels_map.get(qid)
        if not label or label == "unknown":
            missing_qids.add(qid)

    # 3. Fetch missing labels
    if missing_qids:
        logger.info(f"🔍 Need to fetch {len(missing_qids)} new entities")
        logger.info("🌐 Fetching enriched metadata...")

        if intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            missing_qids_log = intermediate_dir / "missing_qids.txt"
            with open(missing_qids_log, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(missing_qids)))
            logger.info(f"📝 Missing QIDs logged to {missing_qids_log}")

        def batch(iterable, size):
            it = iter(iterable)
            while True:
                chunk = list(islice(it, size))
                if not chunk:
                    break
                yield chunk

        total_fetched = 0
        for chunk in tqdm(batch(missing_qids, BATCH_SIZE), total=(len(missing_qids) + BATCH_SIZE - 1) // BATCH_SIZE,
                          desc="Fetching qid labels"):
            batch_results = get_entities_labels_batch(chunk)
            labels_map.update(batch_results)
            total_fetched += len(batch_results)

            if labels_map_path and total_fetched % CHECKPOINT_INTERVAL == 0:
                labels_map_path.parent.mkdir(parents=True, exist_ok=True)
                with open(labels_map_path, "w", encoding="utf-8") as f:
                    json.dump(labels_map, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 [Checkpoint] Saved after {total_fetched} entities to {labels_map_path}")

        logger.info("✅ Labels search complete.")

        # 💾 Final save
        if labels_map_path:
            labels_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(labels_map_path, "w", encoding="utf-8") as f:
                json.dump(labels_map, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Labels map saved to {labels_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update labels map for subgraph RAG.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        '--labels_map_path',
        type=Path,
        default=Path('../../resources/subgraphrag/labels_map.json'),
        help="Path to the labels map file. If it exists, it will be updated with new labels. If not, a new file will be created."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path('../../artifacts/subgraphs/roberta/bw_6_md_2_drt.jsonl'),
        help="Path to the input data file (JSONL format) containing subgraph information."
    )
    parser.add_argument(
        "--intermediate_dir",
        type=Path,
        default=Path("intermediate/subgraphrag/update_labels_map"),
        help="Directory to save intermediate files. If not specified, the intermediate files will be saved in the same directory as the output file, with the name paths.jsonl and scores.jsonl"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/subgraphrag/labels_map_update",
        help="Directory to save the log file."
    )
    args = parser.parse_args()

    log_path = Path(args.log_dir) / f"update_labels_map_{timestamp}.log"
    logger = configure_logger(log_path)

    logger.info(f"📝 Starting update_labels_map with arguments: {args}")
    main(
        labels_map_path=args.labels_map_path,
        input_path=args.input_path,
        intermediate_dir=args.intermediate_dir
    )