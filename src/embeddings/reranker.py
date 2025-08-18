# DEPRECATED
# from src.embeddings.cache import get_cached_embedding
# from src.embeddings.similarity import cosine_similarity
#
#
# def rerank_triplets_by_semantics(entry: dict, embedding_cache: dict) -> dict:
#     """
#     Compute cosine similarity scores between the question and each triplet.
#     Return all triplets sorted by similarity (no top_k truncation).
#     """
#     question = entry["question"]
#     triplets = entry["triplet_texts"]
#
#     q_emb = get_cached_embedding(question, embedding_cache)
#     scores = []
#
#     for text in triplets:
#         triplet_emb = get_cached_embedding(text, embedding_cache)
#         score = cosine_similarity(q_emb, triplet_emb)
#         scores.append((text, score))
#
#     scores.sort(key=lambda x: x[1], reverse=True)
#
#     return {
#         "id": entry["id"],
#         "question": question,
#         "triplets_ranked": [t for t, _ in scores],
#         "scores": [round(s, 4) for _, s in scores],
#     }
