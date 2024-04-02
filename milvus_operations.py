import configparser
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility

def connect_to_milvus():
    """
    Establishes a connection to the Milvus database using details from config.ini.
    """
    cfp = configparser.RawConfigParser()
    cfp.read('config.ini')
    milvus_uri = cfp.get('milvus', 'uri')
    user = cfp.get('milvus', 'user')
    password = cfp.get('milvus', 'password')
    connections.connect("default", uri=milvus_uri, user=user, password=password)
    print("Connected to Milvus.")

def check_and_create_collection(collection_name, dim):
    """
    Checks if a collection exists in Milvus and creates it if not,
    tailored for storing and searching text embeddings from a PDF.

    Parameters:
    - collection_name (str): The name of the collection to check or create.
    - dim (int): The dimension of the embeddings (size of the embedding vector).
    """
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Existing collection '{collection_name}' dropped.")

    # Define the schema for the new collection
    text_id_field = FieldSchema(name="text_id", dtype=DataType.INT64, is_primary=True, description="Unique ID for text segment")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector for text segment")

    schema = CollectionSchema(fields=[text_id_field, embedding_field], auto_id=False, description="Collection for storing text segment embeddings.", enable_dynamic_field=True)
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created for text embeddings.")

def create_index_for_collection(collection_name, field_name, index_params):
    """
    Creates an index for a specified field in the collection.

    Parameters:
    - collection_name (str): The name of the collection.
    - field_name (str): The name of the field to create the index on.
    - index_params (dict): Parameters for the index creation, such as index type and metric type.
    """
    collection = Collection(name=collection_name)
    collection.create_index(field_name=field_name, index_params=index_params)
    print(f"Index created for field '{field_name}' in collection '{collection_name}'.")

def insert_embeddings(collection_name, text_ids, embeddings):
    """
    Inserts text embeddings into the specified Milvus collection.

    Parameters:
    - collection_name (str): The name of the Milvus collection to insert data into.
    - text_ids (list[int]): A list of unique identifiers for each text segment.
    - embeddings (list[list[float]]): A list of embedding vectors for each text segment.
    """
    collection = Collection(name=collection_name)
    entities = [text_ids, embeddings]  # Adjusted to match the expected input format
    insert_result = collection.insert(entities)
    print(f"Inserted {len(text_ids)} embeddings into collection '{collection_name}'.")

def load_collection_into_memory(collection_name):
    """
    Loads the specified collection into memory to prepare for search operations.

    Parameters:
    - collection_name (str): The name of the Milvus collection to be loaded.
    """
    collection = Collection(name=collection_name)
    collection.load()
    print(f"Collection '{collection_name}' loaded into memory.")

def search_embeddings(collection_name, query_embeddings, top_k=5):
    """
    Searches the collection for similar embeddings based on input query embeddings.

    Parameters:
    - collection_name (str): The name of the Milvus collection to search.
    - query_embeddings (list[list[float]]): The query embeddings to search for.
    - top_k (int): Number of nearest embeddings to return.

    Returns:
    - list: A list of search results, each containing IDs of the top_k closest embeddings.
    """
    collection = Collection(name=collection_name)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embeddings,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text_id"]  # Adjust as necessary
    )
    return results


def disconnect_milvus():
    """
    Disconnects from the Milvus database.
    """
    connections.disconnect("default")
    print("Disconnected from Milvus.")
