import json

import requests

from typing import List, Set, cast, TypedDict
from wikimapper import WikiMapper
from pathlib import Path
from datasets import load_dataset

from src.utils.fs import sanitize_filename
from src.utils.logger import logger


class DataStructure(TypedDict):
    qid: str
    titles: List[str]

def expand_titles(titles: List[str]) -> Set[str]:
    """
    Expands each title to include variations:
    - original
    - with underscores replaced by spaces
    - lowercased versions of both
    """
    expanded = set()

    for title in titles:
        with_space = title.replace("_", " ")
        expanded.update({
            title,
            with_space,
            title.lower(),
            with_space.lower(),
        })

    return expanded


def get_articles_from_huggingface(
        data: List[DataStructure],
        output_dir: Path,
) -> List[DataStructure]:
    """
    Fetches articles from the Hugging Face dataset and saves each article's text
    as a separate .txt file named after its title.

    Returns a list of QIDs whose titles were not found in the dataset.
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]
    logger.info(f"✅ Loaded {len(dataset)} Wikipedia articles from Hugging Face.")

    # Precompute expanded titles per QID
    qid_to_titles: dict[str, Set[str]] = {}
    all_titles: Set[str] = set()
    for item in data:
        title_set = set(item["titles"])
        qid_to_titles[item["qid"]] = title_set
        all_titles.update(title_set)

    found_titles: Set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, article in enumerate(dataset):
        title = article["title"]
        content = article["text"]
        if title in all_titles:
            found_titles.add(title)
            filename = sanitize_filename(title)
            file_path = output_dir / f"{filename}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            saved_count += 1

            logger.debug(f"[{idx}] ✅ Saved article: {title}")
        elif idx % 100000 == 0:
            logger.info(f"📦 Processed {idx} articles...")

    logger.info(f"✅ Saved {saved_count} article text files to {output_dir}")

    # Identify QIDs with no matching titles
    not_found: List[DataStructure] = []
    for item in data:
        if not set(item["titles"]) & found_titles:
            not_found.append(item)

    return not_found


def fetch_wikipedia_page(title: str) -> str | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
        "titles": title,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract")
    except Exception as e:
        logger.error(f"🌐 Wikipedia API error for '{title}': {e}")
    return None

def fetch_articles(
        entities: List[str],
        mapper_path: Path,
        output_dir: Path,
        intermediate_dir: Path = None,
) -> None:
    mapper = WikiMapper(str(mapper_path))

    data = []
    not_found_qids = []

    for qid in entities:
        raw_titles = mapper.id_to_titles(qid.strip())
        if not raw_titles:
            logger.warning(f"❌ No titles found for QID {qid}.")
            not_found_qids.append(qid)
            continue

        expanded = expand_titles(raw_titles)
        data.append(cast(DataStructure, {"qid": qid, "titles": expanded}))

    if intermediate_dir:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        nf_qids = intermediate_dir / "02_not_found_with_wikimapper.txt"
        with open(nf_qids, "w", encoding="utf-8") as f:
            for qid in not_found_qids:
                f.write(qid + "\n")
        logger.info(f"💾 Entities that were not found with wikimapper saved to: {nf_qids}")

    not_found_in_hf: List[DataStructure] = get_articles_from_huggingface(data, output_dir)


    if len(not_found_in_hf) == 0:
        logger.info("✅ All titles found in Hugging Face dataset.")
        return


    if intermediate_dir:
        nf_hf_path = intermediate_dir / "03_not_found_in_hf.json"
        with open(nf_hf_path, "w", encoding="utf-8") as f:
            serializable_data = [
                {"qid": item["qid"], "titles": list(item["titles"])}
                for item in not_found_in_hf
            ]
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Entities not found in HF dataset saved to: {nf_hf_path}")

    # Wikipedia API fallback
    logger.warning(f"⚠️ {len(not_found_in_hf)} QIDs still not found after Hugging Face dataset.")
    logger.info("🔄 Fallback to Wikipedia API for not found titles...")

    api_saved = 0
    not_found_with_api: List[DataStructure] = []
    for i, item in enumerate(not_found_in_hf, start=1):
        logger.info(f"🌐 [{i}/{len(not_found_in_hf)}] Trying to fetch QID: {item['qid']}")
        found = False
        for title in item['titles']:
            logger.debug(f"🔍 Trying title: {title}")
            content = fetch_wikipedia_page(title.strip())
            if content:
                filename = sanitize_filename(title)
                file_path = output_dir / f"{filename}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                api_saved += 1
                logger.info(f"✅ Saved article '{title}' to {file_path}")
                found = True
                break # Stop after the first successful title
        if not found:
            logger.warning(f"❌ No valid Wikipedia page found for QID: {item['qid']}")
            not_found_with_api.append(item)

    logger.info(f"🌐 Retrieved {api_saved} articles via Wikipedia API fallback.")

    if len(not_found_with_api) > 0:
        logger.warning(f"⚠️ {len(not_found_with_api)} QIDs still not found after Wikipedia API fallback.")
        if intermediate_dir:
            not_found_api_path = intermediate_dir / "04_not_found_after_api.json"
            with open(not_found_api_path, "w", encoding="utf-8") as f:
                json.dump(not_found_with_api, f, ensure_ascii=False, indent=2)
            logger.warning(f"❌ {len(not_found_with_api)} QIDs not resolved via API. Saved to {not_found_api_path}")

        # Final stats
    final_count = len(list(output_dir.glob("*.txt")))
    logger.info(f"📁 Final total articles saved in '{output_dir}': {final_count}")

