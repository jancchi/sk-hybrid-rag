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

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct"

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
        f"[{i + 1}] Zdroj: {c['title']}\n{c['context']}"
        for i, c in enumerate(contexts)
    )
    return f"ZDROJE:\n{context_block}\n\nOTÁZKA: {question}"


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
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, contexts)},
        ],
        "options": {
            "temperature": 0.1,   # Low temp for factual legal answers
            "num_predict": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
    except httpx.ConnectError:
        logger.error("Cannot reach Ollama. Is it running? `docker compose up -d`")
        raise RuntimeError(
            "Ollama is not reachable at localhost:11434. "
            "Run: docker compose up -d && docker exec sk_ollama ollama pull qwen2.5:7b-instruct"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e}")
        raise
