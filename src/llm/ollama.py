import os
import json
import asyncio
from typing import Optional, Callable, Awaitable

import requests
import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OLLAMA_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("LLM_MODEL_NAME", "llama3.1")


def generate_llm_response(prompt: str, system_prompt: str = "", stream: bool = False, timeout: int = 60) -> str:
    """
    Sends a prompt to the local Ollama server and returns the response.

    Args:
        prompt (str): User prompt to send to the LLM.
        system_prompt (str): Optional system message (e.g., role instruction).
        stream (bool): Whether to stream the response (default: False).
        timeout (int): Timeout in seconds for the request (default: 60).

    Returns:
        str: The LLM-generated response, or error message if failed.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": stream
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ LLM request failed: {e}")


async def agenerate_llm_response(
    prompt: str,
    system_prompt: str = "",
    stream: bool = False,
    timeout: int = 60,
    on_chunk: Optional[Callable[[str], Awaitable[None] | None]] = None,
) -> str:
    """
    Asynchronously sends a prompt to the local Ollama server and returns the response.

    - If stream=False (default): performs a single POST and returns the final text.
    - If stream=True: reads NDJSON chunks from the response stream, optionally
      invoking `on_chunk` with each text piece; returns the concatenated text.

    Args:
        prompt (str): User prompt to send to the LLM.
        system_prompt (str): Optional system message (e.g., role instruction).
        stream (bool): Whether to stream the response incrementally.
        timeout (int): Timeout in seconds for the request.
        on_chunk (Optional[Callable[[str], Awaitable[None] | None]]): Optional async/sync
            callback invoked for each streamed text chunk.

    Returns:
        str: The LLM-generated response text.

    Raises:
        RuntimeError: On network/HTTP/parse errors.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": stream,
    }

    # aiohttp uses asyncio under the hood; this keeps things fully async.
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            if not stream:
                async with session.post(url, json=payload) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(f"❌ LLM request failed: HTTP {resp.status} | body: {text[:500]}")
                    data = await resp.json()
                    return (data.get("response") or "").strip()

            # Streaming mode: Ollama returns NDJSON lines.
            full_text_parts: list[str] = []
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"❌ LLM request failed: HTTP {resp.status} | body: {text[:500]}")

                async for raw_line in resp.content:
                    # Lines may include b'\n'; skip empty lines
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    # Each line should be a JSON object like:
                    # {"response":"...","done":false,...}
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Ignore malformed lines; continue streaming
                        continue

                    part = obj.get("response") or ""
                    if part:
                        full_text_parts.append(part)
                        if on_chunk:
                            maybe_coro = on_chunk(part)
                            if asyncio.iscoroutine(maybe_coro):
                                await maybe_coro

                    if obj.get("done"):
                        break

            return "".join(full_text_parts).strip()

    except aiohttp.ClientError as e:
        raise RuntimeError(f"❌ LLM request failed (network): {e}") from e
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"❌ LLM request timed out after {timeout}s") from e