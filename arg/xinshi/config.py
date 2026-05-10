import os

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "Xinshi_school")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-large")

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

# 多轮对话传入 LLM 的历史条数上限（user/assistant 各算一条）
MAX_HISTORY_MESSAGES = 12
