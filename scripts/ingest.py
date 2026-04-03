"""
SK-QuAD Ingestion Script
========================
Loads SK-QuAD from HuggingFace, deduplicates context paragraphs,
encodes them with BGE-M3 (dense + sparse), and upserts to Qdrant.

Run once before starting the API:
    python scripts/ingest.py

Expected runtime on CPU:  ~45-90 min (22k paragraphs)
Expected runtime on GPU:  ~5-10 min
"""

import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from app.pipeline.embedder import BGEEmbedder
from app.pipeline.retriever import QdrantRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "TUKE-KEMT/SK-QuAD"
BATCH_SIZE = 32       # Reduce to 8-16 if you run out of VRAM
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def load_unique_passages() -> list[dict]:
    """
    Load SK-QuAD train + dev splits and deduplicate by context text.
    
    SK-QuAD structure:
      - Each row: {id, title, context, question, answers}
      - Multiple rows share the same context (different questions)
    
    We index PASSAGES (contexts), not Q&A pairs.
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    seen_contexts: dict[str, dict] = {}

    for split in ["train", "validation"]:
        if split not in dataset:
            logger.warning(f"Split '{split}' not found — skipping.")
            continue

        for row in dataset[split]:
            ctx = row["context"].strip()
            if ctx and ctx not in seen_contexts:
                seen_contexts[ctx] = {
                    "title": row["title"].strip(),
                    "context": ctx,
                }

    passages = list(seen_contexts.values())
    logger.info(f"Unique passages: {len(passages):,}")
    return passages


def ingest():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    passages = load_unique_passages()

    # ------------------------------------------------------------------
    # 2. Init Qdrant collection
    # ------------------------------------------------------------------
    retriever = QdrantRetriever(host=QDRANT_HOST, port=QDRANT_PORT)
    retriever.create_collection()

    existing = retriever.count()
    if existing > 0:
        logger.info(
            f"Collection already has {existing:,} points. "
            "Delete the collection in Qdrant dashboard to re-ingest."
        )
        return

    # ------------------------------------------------------------------
    # 3. Embed + upsert in batches
    # ------------------------------------------------------------------
    embedder = BGEEmbedder(use_fp16=True, batch_size=BATCH_SIZE)

    total = len(passages)
    point_id = 0

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Ingesting batches"):
        batch = passages[start : start + BATCH_SIZE]
        texts = [p["context"] for p in batch]

        dense_vecs, sparse_vecs = embedder.encode(texts)

        ids = list(range(point_id, point_id + len(batch)))
        payloads = [{"title": p["title"], "context": p["context"]} for p in batch]

        retriever.upsert_batch(
            ids=ids,
            dense_vecs=dense_vecs,
            sparse_vecs=sparse_vecs,
            payloads=payloads,
        )

        point_id += len(batch)

    final_count = retriever.count()
    logger.info(f"Ingestion complete. Total points in Qdrant: {final_count:,}")


if __name__ == "__main__":
    ingest()
