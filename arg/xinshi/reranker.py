from sentence_transformers import CrossEncoder
from arg.xinshi.config import *


class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query, docs, top_k):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]
