import argparse
import asyncio
import os

from glob import glob
from lightrag import LightRAG
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from src.lightrag.initialize_rag import initialize_rag
from src.utils.logger import configure_lighrag_logger


def get_processed_files_from_log(log_path: str) -> set[str]:
    processed = set()
    if not os.path.exists(log_path):
        return processed  # no log file yet
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Completed processing file" in line or "⏭️ Already processed, skipping" in line:
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    filename = parts[-1].strip()
                    processed.add(filename)
    return processed

async def insert_all_documents(rag: LightRAG, logger, input_dir: Path, log_path: str = None):
    txt_files = glob(os.path.join(input_dir, '*.txt'))

    processed_files = get_processed_files_from_log(log_path)
    logger.info(f"Found {len(txt_files)} documents to insert")
    logger.info(f"{len(processed_files)} files will be skipped (already processed)")

    success, skipped, failed = 0, 0, 0

    async def process_file(file_path: str):
        nonlocal success, skipped, failed
        filename = os.path.basename(file_path)

        if filename in processed_files:
            logger.info(f"⏭️ Already processed, skipping: {filename}")
            skipped += 1
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"⚠️ Skipping empty file: {filename}")
                skipped += 1
                return

            logger.info(f"📄 Inserting: {filename}")
            await rag.ainsert(input=content, file_paths=filename)
            success += 1

        except Exception as e:
            logger.exception(f"❌ Error inserting {filename}: {e}")
            failed += 1

    await tqdm_asyncio.gather(*(process_file(f) for f in txt_files), desc="📦 Inserting files", ncols=100)

    logger.info("📊 Insertion summary:")
    logger.info(f"✅ Successfully inserted: {success}")
    logger.info(f"⚠️ Skipped (empty): {skipped}")
    logger.info(f"❌ Failed: {failed}")


async def main(
        input_dir: Path,
        working_dir: Path,
):
    rag = None
    logger = configure_lighrag_logger()
    latest_log_path = Path("../../logs/lightrag/lightrag_20250714_180959.log")
    try:
        rag = await initialize_rag(working_dir)
        logger.info(f"LightRAG index initialized at {working_dir}")
        await insert_all_documents(
           rag=rag,
           logger=logger,
           log_path=str(latest_log_path),
           input_dir=input_dir
        )
    except Exception as e:
        logger.exception(f"❌ Error during LightRAG index building: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LightRAG index.")

    parser.add_argument("--working_dir", type=Path, required=True, help="Directory where the LightRAG index will be built.")
    parser.add_argument("--input_dir", type=Path, default=Path("../outputs/processed/articles"), help="Directory containing the input documents.")
    args = parser.parse_args()


    asyncio.run(main(
        input_dir=args.input_dir,
        working_dir=args.working_dir,
    ))


