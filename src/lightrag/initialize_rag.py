import os

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from pathlib import Path
from dotenv import load_dotenv

EMBEDDING_DIM=768
EMBEDDING_MAX_TOKEN_SIZE=8192
MODEL_MAX_TOKEN_SIZE = 32768
load_dotenv()

async def initialize_rag(
    working_dir: Path,
) -> LightRAG:
    """
    Initialize the LightRAG instance with Ollama as the LLM and embedding provider.
    :param working_dir:
    :return:
    """
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL_NAME", 'llama3.1'),
        llm_model_max_async=4,
        llm_model_max_token_size=MODEL_MAX_TOKEN_SIZE,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {
                "num_ctx": MODEL_MAX_TOKEN_SIZE
            }
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("LLM_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                host=os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
            )
        )
    )

    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline

    return rag