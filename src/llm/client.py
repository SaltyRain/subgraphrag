# DEPRECATED
# from dotenv import load_dotenv
# import os
# from llama_index.llms.openai_like import OpenAILike
#
# load_dotenv()
#
# class LLMClient(OpenAILike):
#     """
#     A wrapper class for OpenAI-like LLMs using LlamaIndex's OpenAILike.
#     """
#
#     def __init__(self, model: str = 'llama3', temperature: float = 0.2):
#         api_base = os.getenv("LLM_API_BASE")
#         api_key = os.getenv("LLM_API_KEY")
#
#         if not api_base or not api_key:
#             raise ValueError("Missing required environment variables: LLM_API_KEY and/or LLM_API_BASE")
#
#         super().__init__(
#             model=model,
#             api_base=api_base,
#             api_key=api_key,
#             temperature=temperature,
#             context_window=2048,
#         )