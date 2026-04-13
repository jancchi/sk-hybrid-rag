"""
Ollama Generator
----------------
Sends the reranked context passages + user question to a locally-running
Ollama instance (qwen2.5:7b-instruct) and returns a grounded Slovak answer.

The system prompt explicitly instructs the model to:
  1. Answer only from provided context (no hallucination)
  2. Cite the source number(s) it used
  3. Admit when context is insufficient
"""

import logging
import json
import re
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 540.0

_ollama_client: httpx.AsyncClient | None = None

SYSTEM_PROMPT = """Si odborný právno-finančný asistent pre slovenské právo a dane. \
Odpovedaj vždy v slovenčine. \
Odpovedaj VÝHRADNE na základe poskytnutých zdrojov (označených číslami [1], [2], ...). \
Pri každom tvrdení uveď číslo zdroja v hranatých zátvorkách. \
Ak odpoveď nie je obsiahnutá v zdrojoch, napíš: \
"Na základe dostupných zdrojov neviem odpovedať na túto otázku." \
Nedomýšľaj si ani nedoplňuj informácie mimo poskytnutého kontextu."""


def build_user_message(question: str, contexts: list[dict]) -> str:
    """Format context passages + question into a single user turn."""
    context_block = "\n\n".join(
        f"[{i + 1}] Zdroj: {c['title']}\n{c['context']}" for i, c in enumerate(contexts)
    )
    return f"ZDROJE:\n{context_block}\n\nOTÁZKA: {question}"


def set_ollama_client(client: httpx.AsyncClient | None) -> None:
    """Inject a shared AsyncClient managed by app lifespan."""
    global _ollama_client
    _ollama_client = client


def _build_payload(question: str, contexts: list[dict], stream: bool) -> dict:
    return {
        "model": MODEL_NAME,
        "stream": stream,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, contexts)},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }


def _raise_ollama_runtime_error(e: Exception) -> RuntimeError:
    if isinstance(e, httpx.TimeoutException):
        logger.error(f"Ollama timeout after {OLLAMA_TIMEOUT_SECONDS:.0f} seconds: {e}")
        return RuntimeError(
            "Ollama is taking too long to generate a response "
            f"(timeout after {OLLAMA_TIMEOUT_SECONDS:.0f} seconds). "
            "The model may be overloaded or running slowly. Please try again."
        )
    if isinstance(e, httpx.ConnectError):
        logger.error(f"Cannot reach Ollama. Is it running? {e}")
        return RuntimeError(
            "Ollama is not reachable at localhost:11434. "
            "Run: docker compose up -d && docker exec sk_ollama ollama pull qwen2.5:7b-instruct"
        )
    if isinstance(e, httpx.HTTPStatusError):
        logger.error(f"Ollama HTTP error: {e}")
        return RuntimeError("Ollama returned an HTTP error while generating an answer.")
    if isinstance(e, (KeyError, TypeError, ValueError)):
        logger.error(f"Unexpected Ollama response payload: {e}")
        return RuntimeError("Ollama returned an invalid response payload.")
    logger.error(f"Unexpected Ollama error: {e}")
    return RuntimeError("Unexpected Ollama error while generating an answer.")


def _get_client() -> tuple[httpx.AsyncClient, bool]:
    client = _ollama_client
    if client is not None:
        return client, False
    return httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS), True


def _iter_token_chunks(text: str) -> list[str]:
    """Split text into small token-like chunks while preserving spacing."""
    return re.findall(r"\S+|\s+", text)


async def stream_answer(question: str, contexts: list[dict]) -> AsyncIterator[str]:
    """Stream token chunks from Ollama /api/chat."""
    payload = _build_payload(question=question, contexts=contexts, stream=True)
    client, owns_client = _get_client()

    try:
        async with client.stream(
            "POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                content = message.get("content", "")
                if content:
                    for token_chunk in _iter_token_chunks(content):
                        yield token_chunk
                if chunk.get("done"):
                    break
    except Exception as e:
        raise _raise_ollama_runtime_error(e) from e
    finally:
        if owns_client:
            await client.aclose()


async def generate_answer(question: str, contexts: list[dict]) -> str:
    """
    Call Ollama's /api/chat endpoint (non-streaming).

    Parameters
    ----------
    question : str      — original user question
    contexts : list     — list of dicts with 'title' and 'context' keys
                          (already reranked, best passages first)

    Returns
    -------
    str — the model's answer in Slovak
    """
    payload = _build_payload(question=question, contexts=contexts, stream=False)
    client, owns_client = _get_client()

    try:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except Exception as e:
        raise _raise_ollama_runtime_error(e) from e
    finally:
        if owns_client:
            await client.aclose()
