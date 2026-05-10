import time
from pymilvus import connections
from pymilvus import utility

from arg.xinshi.config import COLLECTION_NAME

for i in range(10):
    try:
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        print("Connected!")
        break
    except Exception as e:
        print("Retrying...", e)
        time.sleep(5)


def drop_collection(name):
    collection_name = name
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

drop_collection(COLLECTION_NAME)
