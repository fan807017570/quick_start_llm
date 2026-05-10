import torch
from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection
from torchgen.api.types import dimVectorT

connections.connect(alias="default", host="localhost", port="19530")

print("Connected to Milvus!")


def get_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description="tensor storage")
    collection = Collection(name="tensor_collection", schema=schema)
    return collection


def convert_to_list():
    tensor = torch.randn(768)
    vector = tensor.detach().cpu().numpy().tolist()
    collection = get_collection();
    collection.insert([[vector]])
    collection.flush()


convert_to_list()
