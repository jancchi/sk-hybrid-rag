"""
BGE-M3 Embedder
---------------
Generates both dense (1024-d) and sparse (lexical weight) vectors
from a single model forward pass. The sparse output acts as a learned
BM25-equivalent, natively supported by Qdrant's sparse vector index.
"""

import logging
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

# BGE-M3 dense vector dimensionality
DENSE_DIM = 1024


class BGEEmbedder:
    """Wraps BAAI/bge-m3 for dual dense+sparse encoding."""

    def __init__(self, use_fp16: bool = True, batch_size: int = 16):
        logger.info("Loading BAAI/bge-m3 — this may take a moment on first run...")
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
        self.batch_size = batch_size
        logger.info("BGE-M3 loaded.")

    def encode(
        self, texts: list[str]
    ) -> tuple[np.ndarray, list[SparseVector]]:
        """
        Encode a list of texts into dense and sparse vectors.

        Returns
        -------
        dense : np.ndarray, shape (N, 1024)
        sparse : list[SparseVector], length N
        """
        output: dict[str, Any] = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense: np.ndarray = output["dense_vecs"]  # (N, 1024)

        sparse: list[SparseVector] = []
        for lexical_weights in output["lexical_weights"]:
            # lexical_weights: dict[str, float] — token_id → importance score
            if not lexical_weights:
                # Fallback: empty sparse vector (shouldn't happen in practice)
                sparse.append(SparseVector(indices=[], values=[]))
                continue
            indices = [int(k) for k in lexical_weights.keys()]
            values = [float(v) for v in lexical_weights.values()]
            sparse.append(SparseVector(indices=indices, values=values))

        return dense, sparse

    def encode_single(self, text: str) -> tuple[np.ndarray, SparseVector]:
        """Convenience wrapper for a single string."""
        dense, sparse = self.encode([text])
        return dense[0], sparse[0]
