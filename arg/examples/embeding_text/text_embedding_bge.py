import torch
import weaviate
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
from weaviate.collections.classes.config import Configure, DataType, Property

model = SentenceTransformer("models/bge-base-en-v1.5", local_files_only=True)


def load_local_model(path):
    tokenizers = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    model.eval


def do_text_embedding(texts):
    return texts, model.encode(
        texts,
        normalize_embeddings=True
    )


def chunk_text(text, chunk_size=100):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def do_text_chunk(docName):
    chunks = []
    with open(docName, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for doc in lines:
            chunks.extend(chunk_text(doc))
    return chunks


def save_vector_to_weaviate(collection_name, text, vectors):
    client = weaviate.connect_to_local(skip_init_checks=True)
    print("weaviate ready?", client.is_ready())
    existing_collections = list(client.collections.list_all(simple=True).keys())
    llm_collection = None
    if collection_name not in existing_collections:
        llm_collection = client.collections.create(
            name=collection_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="text", data_type=DataType.TEXT)
            ]
        )
        print(f"Collection '{collection_name}' created!")
    else:
        llm_collection = client.collections.get(collection_name)

    llm_collection.data.insert(
        properties={
            "text": text
        },
        vector=vectors.tolist()
    )


def do_embeding(docName):
    chunks = do_text_chunk(docName="llm.txt")
    for ch in chunks:
        texts, vectors = do_text_embedding(ch)
        save_vector_to_weaviate("Space_02", texts, vectors)


def query_by_text(queryText, collection_name):
    queryText, queryVector = do_text_embedding(queryText)
    client = weaviate.connect_to_local(skip_init_checks=True)
    llm_collection = client.collections.get(collection_name)
    result = llm_collection.query.near_vector(
        near_vector=queryVector,
        limit=5,
        return_properties=['text']
    )
    print("Query result:")
    for obj in result.objects:
        print(obj.properties)
        client.close();


query_by_text(" Solar Probe Model of the Parker Solar Probe Names ", "Space_02")
