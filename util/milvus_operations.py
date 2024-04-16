import configparser
from pymilvus import MilvusClient, utility, Collection, DataType, FieldSchema, CollectionSchema, utility, connections
from urllib.parse import quote


def connect_to_milvus():
    """
    Establishes a connection to the Milvus database using details from config.ini.
    """
    cfp = configparser.RawConfigParser()
    cfp.read('config.ini')
    
    milvus_uri = cfp.get('milvus', 'uri')
    milvus_token = cfp.get('milvus', 'token')

    connections.connect("default", uri=milvus_uri, token=milvus_token)
    print("Connected to Milvus.")

def check_and_create_collection(collection_name, dim):
    """
    Checks if a collection exists in Milvus and creates it if not,
    tailored for storing and searching text embeddings from a PDF.

    Parameters:
    - collection_name (str): The name of the collection to check or create.
    - dim (int): The dimension of the embeddings (size of the embedding vector).
    """
    # Check if the collection already exists
    if not utility.has_collection(collection_name):
        # Define the schema for the new collection
        text_id_field = FieldSchema(name="text_id", dtype=DataType.INT64, is_primary=True, description="Unique ID for text segment")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="Embedding vector for text segment")

        schema = CollectionSchema(fields=[text_id_field, embedding_field], description="Collection for storing text segment embeddings.")
        
        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection.name}' created for text embeddings.")
        return True
    else:
        print(f"Collection '{collection_name}' already exists.")
        return False

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

def milvus_insert(collection_name, text_ids, embeddings):
    """
    Inserts embeddings along with their identifiers into a specified Milvus collection.
    This function is intended for storing vectorized text data from pages of PDFs, where each text segment's ID corresponds to a unique identifier.

    Parameters:
    - collection_name (str): Name of the Milvus collection to which the data will be inserted.
    - text_ids (list[int]): List of integer identifiers corresponding to text segments. These identifiers should be unique to ensure accurate retrieval.
    - embeddings (list[list[float]]): List of embeddings, where each embedding is a list of floats representing the vectorized form of a text segment.
    """
    collection = Collection(name=collection_name)
    entities = [
        text_ids,  # Primary key field
        embeddings,  # Vector embeddings field
    ]
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

def search_embeddings(collection_name, query_embeddings, top_k=3):
    collection = Collection(name=collection_name)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embeddings,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text_id"]
    )

    # Convert results to a serializable format
    processed_results = []
    for result in results:
        for hit in result:
            processed_results.append({
                "score": hit.score,
                "id": hit.id,
                "text_id": hit.entity.get("text_id")
            })
    
    return processed_results


def disconnect_milvus():
    """
    Disconnects from the Milvus database.
    """
    connections.disconnect("default")
    print("Disconnected from Milvus.")

def initialize_milvus_system(collection_name, dim=1536):
    """
    Initializes the connection to Milvus, ensures the specified collection exists,
    and creates an index if one does not already exist.

    Parameters:
    - collection_name (str): The name of the Milvus collection.
    - dim (int): Dimension of the embeddings, defaults to 1536.
    """
    connect_to_milvus()
    created_new_collection = check_and_create_collection(collection_name, dim)

    # If a new collection was created, also create an index for it
    if created_new_collection:
        field_name = "embedding"  # This should match the field defined in your schema
        index_params = {
            "index_type": "IVF_FLAT",  # The index type you want to use
            "metric_type": "L2",       # The metric type (e.g., L2 distance)
            "params": {"nlist": 100}   # The index creation parameters
        }
        create_index_for_collection(collection_name, field_name, index_params)


