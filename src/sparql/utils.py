import requests
import os
from src.utils.logger import logger

VERBOSE = os.getenv("VERBOSE")

# Constants
DEFAULT_LOCAL_SPARQL = "http://localhost:1234/api/endpoint/sparql"
REMOTE_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
#
# @lru_cache(maxsize=10_000)
# def get_entity_label(
#         qid: str,
#         lang: str = "en",
#         source: Literal["api", "sparql"] = "sparql",
#         offline: bool = False,
#         local_sparql_url: Optional[str] = DEFAULT_LOCAL_SPARQL
# ) -> str:
#     """
#     Fetch the label of a Wikidata entity using SPARQL or API, with fallback between methods.
#
#     Args:
#         qid (str): The QID or PID of the entity (e.g., "Q42", "P69").
#         lang (str): Language code (default is "en").
#         source (str): Data source to use: 'api'  or 'sparql' (default).
#         offline (bool): If True, no network requests are made.
#         local_sparql_url (str): URL of local SPARQL endpoint (optional).
#
#     Returns:
#         str: Human-readable label of the entity, or the QID if not found.
#     """
#
#     if offline:
#         return qid
#
#     def fetch_from_sparql(qid: str, endpoint: str, label_source: str) -> Optional[str]:
#         if endpoint == REMOTE_SPARQL:
#             time.sleep(0.1)
#
#         query = f"""
#         SELECT ?label WHERE {{
#           wd:{qid} rdfs:label ?label.
#           FILTER (lang(?label) = "{lang}")
#         }} LIMIT 1
#         """
#
#         headers = {
#             "Accept": "application/sparql-results+json",
#             "User-Agent": os.getenv("USER_AGENT"),
#         }
#
#         try:
#             response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=5)
#             response.raise_for_status()
#             results = response.json().get("results", {}).get("bindings", [])
#             if results:
#                 label = results[0]["label"]["value"]
#                 if VERBOSE:
#                     logger.info(f"[SPARQL:{label_source}] {qid} → {label}")
#
#                 return label
#
#         except Exception as e:
#             logger.info(f"[SPARQL error] {endpoint} failed for {qid}: {e}")
#         return None
#
#     def fetch_from_api(qid: str) -> Optional[str]:
#         params = {
#             "action": "wbgetentities",
#             "ids": qid,
#             "languages": lang,
#             "props": "labels",
#             "format": "json"
#         }
#         try:
#             response = requests.get(WIKIDATA_API, params=params, timeout=5)
#             response.raise_for_status()
#             data = response.json()
#             if 'entities' in data and qid in data['entities']:
#                 labels = data['entities'][qid].get("labels", {})
#                 if lang in labels:
#                     label =  labels[lang]["value"]
#                     if VERBOSE:
#                         logger.info(f"[API] {qid} → {label}")
#                     return label
#         except Exception as e:
#             logger.info(f"[API error] Failed to fetch {qid}: {e}")
#         return None
#
#     # Retrieval order based on primary source
#     if source == "sparql":
#         return (
#             fetch_from_sparql(qid, local_sparql_url, 'local')
#             or fetch_from_sparql(qid, REMOTE_SPARQL, 'remote')
#             or fetch_from_api(qid)
#             or qid
#         )
#
#     elif source == "api":
#         return (
#             fetch_from_api(qid)
#             or fetch_from_sparql(qid, local_sparql_url, 'local')
#             or fetch_from_sparql(qid, REMOTE_SPARQL, 'remote')
#             or qid
#         )
#
#     else:
#         raise ValueError("Invalid source. Choose 'sparql' or 'api'.")
#
#
#
#
# @lru_cache(maxsize=10_000)
# def get_entity_enriched_data(
#     qid: str,
#     lang: str = "en",
#     local_sparql_url: str = DEFAULT_LOCAL_SPARQL,
# ) -> EnrichedEntity:
#     """
#     Fetches enriched metadata (label, types, locations) for the given QID using local SPARQL endpoint.
#     """
#
#     query = f"""
#     SELECT ?label ?typeLabel ?locationLabel WHERE {{
#       OPTIONAL {{ wd:{qid} rdfs:label ?label FILTER(lang(?label) = "{lang}") }}
#
#       OPTIONAL {{
#         wd:{qid} wdt:P31 ?type .
#         ?type rdfs:label ?typeLabel FILTER(lang(?typeLabel) = "{lang}")
#       }}
#
#       OPTIONAL {{
#         wd:{qid} wdt:P279 ?type .
#         ?type rdfs:label ?typeLabel FILTER(lang(?typeLabel) = "{lang}")
#       }}
#
#       OPTIONAL {{
#         VALUES ?locProp {{ wdt:P17 wdt:P131 wdt:P159 }}
#         wd:{qid} ?locProp ?location .
#         ?location rdfs:label ?locationLabel FILTER(lang(?locationLabel) = "{lang}")
#       }}
#     }}
#     """
#
#     headers = {
#         "Accept": "application/sparql-results+json",
#         "User-Agent": os.getenv("USER_AGENT") or "LocalRAG/1.0"
#     }
#
#     try:
#         response = requests.get(local_sparql_url, params={"query": query}, headers=headers, timeout=10)
#         response.raise_for_status()
#         bindings = response.json().get("results", {}).get("bindings", [])
#     except Exception as e:
#         logger.info(f"[SPARQL error] {local_sparql_url} failed for {qid}: {e}")
#         return {
#             "id": qid,
#             "label": None,
#             "types": [],
#             "locations": [],
#         }
#
#     label = None
#     types = set()
#     locations = set()
#
#     for b in bindings:
#         if "label" in b:
#             label = b["label"]["value"]
#         if "typeLabel" in b:
#             types.add(b["typeLabel"]["value"])
#         if "locationLabel" in b:
#             locations.add(b["locationLabel"]["value"])
#
#     return {
#         "id": qid,
#         "label": label,
#         "types": sorted(types),
#         "locations": sorted(locations),
#     }
#
#
# def get_entity_enriched_data_safe(
#     qid: str,
#     enriched_so_far: Dict[str, EnrichedEntity],
# ) -> EnrichedEntity:
#     if qid in enriched_so_far:
#         return enriched_so_far[qid]
#
#     try:
#         result = get_entity_enriched_data(qid)
#         if result.get("label"):
#             return result
#     except Exception as e:
#         logger.info(f"❌ Failed to fetch {qid}: {e}")
#
#     logger.info(f"⚠️ Skipping {qid} due to error.")
#     return {
#         "id": qid,
#         "label": None,
#         "types": [],
#         "locations": [],
#     }



def get_entities_labels_batch(
        qids: list[str],
        lang: str = "en",
        local_sparql_url: str = DEFAULT_LOCAL_SPARQL,
) -> dict[str, str]:
    """
    Fetch labels for a batch of QIDs using a single SPARQL query.

    Returns:
        Dict where key = QID, value = label (or "unknown" if not found).
    """
    values_clause = " ".join(f"wd:{qid}" for qid in qids)
    logger.info(f"Fetching labels for: {qids[:3]}... (total {len(qids)})")

    query = f"""
    SELECT ?entity ?label WHERE {{
      VALUES ?entity {{ {values_clause} }}
      ?entity rdfs:label ?label .
      FILTER(lang(?label) = "{lang}")
    }}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": os.getenv("USER_AGENT") or "LocalRAG/1.0"
    }

    try:
        response = requests.get(local_sparql_url, params={"query": query}, headers=headers, timeout=60)
        response.raise_for_status()
        bindings = response.json().get("results", {}).get("bindings", [])
    except Exception as e:
        logger.warning(f"[SPARQL ERROR] Failed for {qids[:3]}...: {e}")
        return {qid: "unknown" for qid in qids}

    results: dict[str, str] = {qid: "unknown" for qid in qids}

    for b in bindings:
        url = b.get("entity", {}).get("value", "")
        qid = url.rsplit("/", 1)[-1]
        label = b.get("label", {}).get("value", "unknown")
        if qid in results:
            results[qid] = label

    return results
