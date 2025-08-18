# DEPRECATED
# import os
#
# import requests
# import dotenv
#
# dotenv.load_dotenv()
#
# def get_embedding(text: str, model: str = "nomic-embed-text:latest") -> list[float]:
#     base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
#     url = f"{base_url.rstrip('/')}/api/embeddings"
#
#     response = requests.post(
#         url,
#         json={"model": model, "prompt": text}
#     )
#     response.raise_for_status()
#     return response.json()["embedding"]