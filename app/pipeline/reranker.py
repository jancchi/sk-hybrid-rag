"""
Cross-Encoder Reranker
----------------------
Uses a multilingual cross-encoder to do deep semantic verification of
the top-K candidates from hybrid retrieval.

Unlike bi-encoders (where query and passage are encoded independently),
a cross-encoder sees query+passage jointly — enabling token-level
attention across both. This is significantly more accurate but slower,
which is why it only runs on the already-filtered top-K.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - Trained on MS MARCO in 26 languages (includes Slovak-adjacent languages)
  - Distilled from a larger model — fast enough for real-time CPU inference
  - Better than ms-marco-MiniLM-L-6-v2 for non-English queries
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class RankedPassage:
    original_index: int  # index into the input list
    title: str
    context: str
    cross_score: float  # raw logit from cross-encoder


class CrossEncoderReranker:
    def __init__(self):
        logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        logger.info("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        passages: list[dict],  # each dict has 'title' and 'context' keys
        top_k: int = 5,
    ) -> list[RankedPassage]:
        """
        Score each (query, passage) pair and return top_k sorted by score.

        Parameters
        ----------
        query    : User's Slovak question
        passages : List of dicts from retriever (title, context)
        top_k    : How many to return after reranking
        """
        if not passages:
            return []

        # Build (query, passage_text) pairs for the cross-encoder
        pairs = [[query, p["context"]] for p in passages]
        scores: list[float] = self.model.predict(pairs).tolist()

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RankedPassage(
                original_index=idx,
                title=passages[idx].get("title", ""),
                context=passages[idx]["context"],
                cross_score=score,
            )
            for idx, score in ranked
        ]
