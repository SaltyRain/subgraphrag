# DEPRICATED
# import re
#
# def extract_used_triplets_from_answer(answer: str) -> list[str]:
#     """
#     Extract supporting triplets mentioned in the LLM's answer.
#     Looks for lines following a phrase like "The facts that support this answer are:".
#
#     Returns a list of triplet strings.
#     """
#     used = []
#     support_block = re.findall(r"The facts that support.*?:\s*(.*?)\Z", answer, re.DOTALL | re.IGNORECASE)
#     if support_block:
#         lines = support_block[0].strip().splitlines()
#         for line in lines:
#             line = line.strip("-• ").strip()
#             if line.count("–") == 2:  # Simple heuristic to match triplets
#                 used.append(line)
#     return used
#
#
# def is_fully_grounded(used_triplets: list[str], all_triplets: list[dict]) -> bool:
#     original_set = {
#         f"{t['subject']['label']} – {t['predicate']['label']} – {t['object']['label']}"
#         for t in all_triplets
#     }
#     return all(t in original_set for t in used_triplets)
