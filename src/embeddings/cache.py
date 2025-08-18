# DEPRECATED
#
# import json
# from pathlib import Path
#
# from src.embeddings.client import get_embedding
#
#
# def load_embedding_cache(path: Path) -> dict[str, list[float]]:
#     cache = {}
#     if not path.exists():
#         return cache
#
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 obj = json.loads(line)
#                 text = obj["text"]
#                 embedding = obj["embedding"]
#                 cache[text] = embedding
#     except Exception as e:
#         print(f"⚠️ Failed to load embedding cache from {path}: {e}")
#
#     return cache
#
#
# def save_embedding_cache(cache: dict[str, list[float]], path: Path):
#     try:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "w", encoding="utf-8") as f:
#             for text, embedding in cache.items():
#                 json.dump({"text": text, "embedding": embedding}, f, ensure_ascii=False)
#                 f.write("\n")
#         print(f"💾 Embedding cache saved to {path}")
#     except Exception as e:
#         print(f"❌ Failed to save embedding cache to {path}: {e}")
#
#
#
# def get_cached_embedding(text: str, cache: dict[str, list[float]]) -> list[float]:
#     if text in cache:
#         return cache[text]
#     emb = get_embedding(text)
#
#     # Update cache
#     cache[text] = emb
#     return emb
#

