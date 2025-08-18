import json
from pathlib import Path
from typing import Any, Union

from pandas import DataFrame

from src.utils.logger import logger
import re

def save_intermediate(data: Any, path: Path, label: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"{label} saved to {path}")


def ensure_directory(path: Path, label: str):
    if not path.exists():
        logger.info(f"Creating {label} directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Using existing {label} directory: {path}")

def sanitize_filename(title: str) -> str:
    """
    Converts a title string into a safe filename.
    Removes invalid characters and replaces spaces with underscores.
    """
    title = Path(title).name
    title = re.sub(r"[^\w\s\-]", "", title)
    title = title.strip().replace(" ", "_")
    return title[:100]

def write_df_to_csv(path: Union[str, Path], df: DataFrame) -> None:
    """Write a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_txt(path: Union[str, Path], text: str) -> None:
    """Write a string to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)