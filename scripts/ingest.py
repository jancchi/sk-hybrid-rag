"""
SK-QuAD Ingestion Script
========================
Loads corpus from TUKE-KEMT/retrieval-skquad (corpus.jsonl),
encodes them with BGE-M3 (dense + sparse), and upserts to Qdrant.

Run once before starting the API:
    python scripts/ingest.py

Expected runtime on CPU:  ~45-90 min (~70k documents)
Expected runtime on GPU:  ~5-10 min
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from huggingface_hub import hf_hub_download

# DO NOT import BGEEmbedder or QdrantRetriever here — they depend on torch
# which must be configured FIRST in ingest() before model initialization

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME = "TUKE-KEMT/retrieval-skquad"
BATCH_SIZE = 8
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def _build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SK-QuAD corpus into Qdrant")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--host", type=str, default=QDRANT_HOST)
    parser.add_argument("--port", type=int, default=QDRANT_PORT)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for embedding (recommended only with GPU).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=int(os.getenv("INGEST_NUM_THREADS", "0")),
        help="Torch intra-op threads (0 keeps torch default).",
    )
    return parser.parse_args()


def _configure_torch_threads(num_threads: int) -> None:
    if num_threads <= 0:
        return
    try:
        import torch

        torch.set_num_threads(num_threads)
        if num_threads > 1:
            torch.set_num_interop_threads(max(1, num_threads // 2))
        logger.info(
            "Configured torch threads: num_threads=%s, num_interop_threads=%s",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
        )
    except Exception as exc:
        logger.warning("Could not set torch thread count: %s", exc)


def load_unique_passages() -> list[dict]:
    """
    Load SK-QuAD corpus from TUKE-KEMT/retrieval-skquad.

    Corpus structure (corpus.jsonl):
       - Each line: {_id, title, text}

    We index documents (texts), one per line.
    """
    logger.info(f"Loading corpus from: {DATASET_NAME}")

    # Download corpus.jsonl from HuggingFace
    corpus_file = hf_hub_download(
        repo_id=DATASET_NAME,
        filename="corpus.jsonl",
        repo_type="dataset",
    )

    passages = []
    with open(corpus_file, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            passages.append(
                {
                    "id": doc.get("_id", ""),
                    "title": doc.get("title", "").strip(),
                    "context": doc.get("text", "").strip(),
                }
            )

    logger.info(f"Loaded passages: {len(passages):,}")
    return passages


def ingest() -> None:
    args = _build_parser()
    _configure_torch_threads(args.num_threads)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    logger.info(
        "Ingest config: batch_size=%s fp16=%s qdrant=%s:%s",
        args.batch_size,
        args.fp16,
        args.host,
        args.port,
    )

    # Import after torch configuration
    from app.pipeline.embedder import BGEEmbedder
    from app.pipeline.retriever import QdrantRetriever

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    passages = load_unique_passages()

    # ------------------------------------------------------------------
    # 2. Init Qdrant collection
    # ------------------------------------------------------------------
    retriever = QdrantRetriever(host=args.host, port=args.port)
    retriever.create_collection()

    existing = retriever.count()
    total = len(passages)

    if existing >= total:
        logger.info(
            "Collection already has %s points (>= total passages %s). Nothing to ingest.",
            f"{existing:,}",
            f"{total:,}",
        )
        return

    if existing > 0:
        logger.info(
            "Resuming ingestion from existing count: %s/%s points",
            f"{existing:,}",
            f"{total:,}",
        )
    else:
        logger.info("Starting fresh ingestion for %s passages", f"{total:,}")

    # ------------------------------------------------------------------
    # 3. Embed + upsert in batches
    # ------------------------------------------------------------------
    embedder = BGEEmbedder(use_fp16=args.fp16, batch_size=args.batch_size)

    point_id = existing
    start_idx = existing

    embed_total_s = 0.0
    upsert_total_s = 0.0

    started_at = time.perf_counter()

    for start in tqdm(
        range(start_idx, total, args.batch_size), desc="Ingesting batches"
    ):
        batch = passages[start : start + args.batch_size]
        texts = [p["context"] for p in batch]

        t0 = time.perf_counter()
        dense_vecs, sparse_vecs = embedder.encode(texts)
        embed_total_s += time.perf_counter() - t0

        ids = list(range(point_id, point_id + len(batch)))
        payloads = [{"title": p["title"], "context": p["context"]} for p in batch]

        t1 = time.perf_counter()
        retriever.upsert_batch(
            ids=ids,
            dense_vecs=dense_vecs,
            sparse_vecs=sparse_vecs,
            payloads=payloads,
        )
        upsert_total_s += time.perf_counter() - t1

        point_id += len(batch)

        if point_id % (args.batch_size * 25) == 0:
            done = point_id - existing
            elapsed = time.perf_counter() - started_at
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Progress: %s/%s new points, avg throughput %.2f passages/s",
                f"{done:,}",
                f"{total - existing:,}",
                rate,
            )

    final_count = retriever.count()
    total_elapsed_s = time.perf_counter() - started_at
    logger.info(
        "Ingestion complete. Total points in Qdrant: %s | elapsed=%.1fs embed=%.1fs upsert=%.1fs",
        f"{final_count:,}",
        total_elapsed_s,
        embed_total_s,
        upsert_total_s,
    )


if __name__ == "__main__":
    ingest()
