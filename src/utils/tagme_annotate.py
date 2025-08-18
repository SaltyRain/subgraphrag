import os
import requests

from dotenv import load_dotenv
from typing import List

from src.types import TagMeAnnotation

load_dotenv()

# NOTE: For more information on the TagMe API, see:
# https://sobigdata.d4science.org/web/tagme/tagme-help
def tagme_annotate(
    text: str,
) -> List[TagMeAnnotation]:
    """
    Annotate the input text using the TagMe API.
    :param text: Question or sentence to annotate.
    :return: List of TagMeAnnotation objects passing the threshold.
    """
    params = {
        "text": text,
        "gcube-token": os.getenv("TAGME_API_KEY"),
        "lang": "en",
        "include_abstract": "true",
        "include_categories": "true",
        "include_all_spots": "true",

    }
    response  = requests.get(os.getenv("TAGME_API_URL"), params=params)

    if response.status_code != 200:
        print(f"TagMe API error: {response.status_code} - {response.text}")
        return []

    annotations: List[TagMeAnnotation] = response.json().get("annotations", [])

    return annotations