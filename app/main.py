"""
SK Legal Assistant — FastAPI Backend
=====================================
Single endpoint: POST /ask

Pipeline per request:
  1. Embed query (BGE-M3 dense + sparse)        ~200ms
  2. Hybrid search in Qdrant (RRF fusion)        ~50ms
  3. Cross-encoder rerank top-20 → top-5        ~400ms
  4. Ollama generation (qwen2.5:7b-instruct)    ~5-15s

Startup: models are loaded once and reused across requests (lifespan).
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.embedder import BGEEmbedder
from app.pipeline.retriever import QdrantRetriever
from app.pipeline.reranker import CrossEncoderReranker
from app.pipeline.generator import generate_answer
from app.schemas import AskRequest, AskResponse, SourcePassage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup
# ---------------------------------------------------------------------------

class PipelineState:
    embedder: BGEEmbedder = None
    retriever: QdrantRetriever = None
    reranker: CrossEncoderReranker = None

state = PipelineState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models before the server starts accepting requests."""
    logger.info("=== Loading pipeline models ===")
    state.embedder = BGEEmbedder(use_fp16=True)
    state.retriever = QdrantRetriever(host="localhost", port=6333)
    state.reranker = CrossEncoderReranker()

    # Sanity check: make sure the collection exists and has data
    count = state.retriever.count()
    if count == 0:
        logger.warning(
            "Qdrant collection is empty! Run: python scripts/ingest.py"
        )
    else:
        logger.info(f"Qdrant collection has {count:,} passages ready.")

    logger.info("=== Pipeline ready ===")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SK Legal Assistant",
    description="RAG pipeline over SK-QuAD: BGE-M3 hybrid retrieval + cross-encoder rerank + Qwen2.5",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_passages": state.retriever.count() if state.retriever else 0,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Full RAG pipeline:
      query → embed → hybrid retrieve → cross-encoder rerank → LLM generate
    """
    if state.embedder is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    question = request.question.strip()

    # ------------------------------------------------------------------
    # Stage 1: Embed query
    # ------------------------------------------------------------------
    dense_vec, sparse_vec = state.embedder.encode_single(question)

    # ------------------------------------------------------------------
    # Stage 2: Hybrid retrieval (BGE-M3 dense + sparse → Qdrant RRF)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    candidates = state.retriever.hybrid_search(
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        top_k=request.top_k_retrieve,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages found. Is the collection populated?",
        )

    # ------------------------------------------------------------------
    # Stage 3: Cross-encoder reranking
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    passage_dicts = [
        {"title": c.title, "context": c.context} for c in candidates
    ]
    reranked = state.reranker.rerank(
        query=question,
        passages=passage_dicts,
        top_k=request.top_k_rerank,
    )
    rerank_ms = (time.perf_counter() - t1) * 1000

    top_contexts = [
        {"title": r.title, "context": r.context} for r in reranked
    ]

    # ------------------------------------------------------------------
    # Stage 4: LLM generation
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    answer = await generate_answer(question=question, contexts=top_contexts)
    generation_ms = (time.perf_counter() - t2) * 1000

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    sources = [
        SourcePassage(
            title=r.title,
            context=r.context,
            score=round(r.cross_score, 4),
        )
        for r in reranked
    ]

    logger.info(
        f"[/ask] retrieve={retrieval_ms:.0f}ms rerank={rerank_ms:.0f}ms "
        f"generate={generation_ms:.0f}ms passages={len(candidates)}"
    )

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        generation_ms=round(generation_ms, 1),
    )
