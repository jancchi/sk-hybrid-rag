# SK Právny a Finančný Asistent

RAG pipeline over SK-QuAD: multilingual hybrid retrieval + cross-encoder reranking + local LLM generation.

## Architecture

```
User Query (Slovak)
       │
       ▼
┌─────────────────────────────────────────┐
│  Stage 1: Hybrid Retrieval              │
│                                         │
│  BGE-M3 → dense vec (1024d)  ──┐        │
│  BGE-M3 → sparse vec         ──┼─▶ RRF │  top-20
│                                         │
│  Both stored in Qdrant (single coll.)   │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Stage 2: Cross-Encoder Reranking       │
│                                         │
│  mmarco-mMiniLMv2-L12-H384-v1          │  top-5
│  (26-language, CPU-friendly)            │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Stage 3: Generation                    │
│                                         │
│  Ollama: qwen2.5:7b-instruct            │
│  Grounded Slovak answer + source refs   │
└─────────────────────────────────────────┘
```

## Stack

| Component       | Choice                                   |
|-----------------|------------------------------------------|
| Embeddings      | `BAAI/bge-m3` (dense 1024d + sparse)    |
| Vector DB       | Qdrant (Docker)                          |
| Retrieval       | Qdrant Hybrid Search → RRF               |
| Reranker        | `cross-encoder/mmarco-mMiniLMv2-L12...` |
| Generator       | Ollama `qwen2.5:7b-instruct`            |
| Backend         | FastAPI                                  |
| Frontend        | Streamlit                                |
| Dataset         | SK-QuAD corpus (`TUKE-KEMT/retrieval-skquad`)  |

---

## Setup

### 1. Start infrastructure

```bash
docker compose up -d
```

Wait ~10 seconds for Qdrant to be healthy.

### 2. Pull the LLM

```bash
docker exec sk_ollama ollama pull qwen2.5:7b-instruct
```

This downloads ~4.7 GB. Run once.

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingest SK-QuAD into Qdrant

```bash
python scripts/ingest.py --batch-size 8 --num-threads 8
```

Downloads corpus from `TUKE-KEMT/retrieval-skquad`, encodes ~70k documents with BGE-M3,
and upserts to Qdrant. This is the slow step (~45-90 min on CPU, ~5-10 min on GPU).

CPU note: `ingest.py` now defaults to fp32 (better for most CPU-only systems). Enable fp16 only when using GPU by adding `--fp16`.

If ingestion is interrupted, re-running `ingest.py` resumes from the current Qdrant count.

Progress is shown with a tqdm bar. Safe to re-run — skips if collection already populated.

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

Models load at startup (~30-60s). Check the logs for "Pipeline ready".

### 6. Start the frontend

In a new terminal:

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`.

---

## API

### `POST /ask`

```json
{
  "question": "Aké sú podmienky pre daňový bonus na dieťa?",
  "top_k_retrieve": 20,
  "top_k_rerank": 5
}
```

Response:

```json
{
  "question": "...",
  "answer": "Podľa zdroja [1], daňový bonus...",
  "sources": [
    {"title": "Zákon o dani z príjmov", "context": "...", "score": 0.912}
  ],
  "retrieval_ms": 48.3,
  "rerank_ms": 390.1,
  "generation_ms": 8420.5
}
```

### `GET /health`

Returns Qdrant passage count and API status.

---

## Notes

- **GPU**: Uncomment the `deploy.resources` block in `docker-compose.yml` for Ollama GPU passthrough (requires `nvidia-container-toolkit`).
- **Cross-encoder speed**: ~400ms on a modern CPU for top-20 candidates. Acceptable for local use.
- **BGE-M3 VRAM**: ~2.5 GB in fp16. Falls back to CPU automatically if no GPU.
- **Re-ingestion**: To wipe and re-ingest, delete the Qdrant collection via `http://localhost:6333/dashboard` and re-run `ingest.py`.
