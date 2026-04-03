"""
Qdrant Hybrid Retriever
-----------------------
Runs dense and sparse searches in parallel via Qdrant's `Prefetch` API,
then fuses the result lists with Reciprocal Rank Fusion (RRF).
Both vector types live in a single collection — no dual-index overhead.
"""

import logging
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sk_quad"
DENSE_DIM = 1024


@dataclass
class RetrievedPassage:
    id: int
    title: str
    context: str
    score: float  # RRF fused score from Qdrant


class QdrantRetriever:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """Create the SK-QuAD collection with dense + sparse vector config."""
        if self.client.collection_exists(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        logger.info(f"Created collection '{COLLECTION_NAME}'.")

    def upsert_batch(
        self,
        ids: list[int],
        dense_vecs: np.ndarray,
        sparse_vecs: list[SparseVector],
        payloads: list[dict],
    ) -> None:
        """Insert a batch of pre-encoded points."""
        points = [
            PointStruct(
                id=idx,
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": sparse_vecs[i],
                },
                payload=payloads[i],
            )
            for i, idx in enumerate(ids)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def count(self) -> int:
        return self.client.count(COLLECTION_NAME).count

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        dense_vec: np.ndarray,
        sparse_vec: SparseVector,
        top_k: int = 20,
    ) -> list[RetrievedPassage]:
        """
        Two-arm hybrid search fused with RRF.

        Stage 1a — Dense:  cosine similarity over 1024-d embeddings.
        Stage 1b — Sparse: dot product over BGE-M3 lexical weights.
        Fusion:    Qdrant's built-in RRF merges both ranked lists.
        """
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec.tolist(),
                    using="dense",
                    limit=top_k * 3,  # over-fetch for better fusion
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using="sparse",
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        passages: list[RetrievedPassage] = []
        for hit in results.points:
            payload = hit.payload or {}
            passages.append(
                RetrievedPassage(
                    id=hit.id,
                    title=payload.get("title", ""),
                    context=payload.get("context", ""),
                    score=hit.score,
                )
            )
        return passages
