import heapq
import logging
import time

from sentence_transformers import CrossEncoder
from arg.xinshi.config import *

log = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self):
        self.batch_size = RERANK_BATCH_SIZE
        self.max_length = RERANK_MAX_LENGTH
        self.backend = RERANK_BACKEND
        try:
            self.model = CrossEncoder(
                RERANK_MODEL,
                max_length=self.max_length,
                backend=self.backend,
            )
        except Exception:
            log.warning(
                "reranker backend=%s unavailable, fallback to torch", self.backend,
                exc_info=True,
            )
            self.backend = "torch"
            self.model = CrossEncoder(
                RERANK_MODEL,
                max_length=self.max_length,
                backend=self.backend,
            )
        log.info(
            "reranker ready model=%s backend=%s batch_size=%d max_length=%d",
            RERANK_MODEL,
            self.backend,
            self.batch_size,
            self.max_length,
        )

    def rerank(self, query, docs, top_k):
        if not docs or top_k <= 0:
            return []
        if len(docs) == 1:
            return docs[:1]
        if top_k >= len(docs):
            # 召回数不大于 top_k 时，保留相似检索原排序，避免额外 rerank 计算。
            return docs

        t0 = time.perf_counter()
        all_scores = []
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i:i + self.batch_size]
            pairs = [(query, d.page_content) for d in batch_docs]
            batch_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            if hasattr(batch_scores, "tolist"):
                all_scores.extend(batch_scores.tolist())
            else:
                all_scores.extend(list(batch_scores))

        top_k = min(top_k, len(docs))
        ranked = heapq.nlargest(
            top_k,
            zip(docs, all_scores),
            key=lambda x: x[1]
        )
        log.info(
            "rerank done docs=%d top_k=%d elapsed=%.3fs batch_size=%d max_length=%d",
            len(docs),
            top_k,
            time.perf_counter() - t0,
            self.batch_size,
            self.max_length,
        )
        return [doc for doc, _ in ranked[:top_k]]
