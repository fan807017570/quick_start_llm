from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from arg.xinshi.config import *


def get_vectorstore():
    # 与 ingest.py 使用同一套 HuggingFaceEmbeddings，避免检索与入库向量不一致
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )

