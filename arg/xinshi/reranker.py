import os

from sentence_transformers import CrossEncoder
from arg.xinshi.config import *


class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)
        self.batch_size = max(1, int(os.environ.get("RERANK_BATCH_SIZE", "32")))

    def rerank(self, query, docs, top_k):
        if not docs or top_k <= 0:
            return []

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

        ranked = sorted(
            zip(docs, all_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]
