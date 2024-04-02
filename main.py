from extract_text import extract_pdf_text
import configparser
import openai

# Read the API key from the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('openai', 'api_key')

# Set the OpenAI API key
openai.api_key = openai_api_key
from embed_text import generate_text_embeddings
from milvus_operations import connect_to_milvus, check_and_create_collection, insert_embeddings, search_embeddings, disconnect_milvus, load_collection_into_memory, create_index_for_collection

query = 'The responsibility of government is to “sacredly guard” the rights of property for the prosperity of the community.'

if __name__ == "__main__":
    pdf_path = "data/supremecourt_landmarkcases 01.pdf"
    collection_name = "landmark_cases"
    dim = 1536  # Dimension of your embeddings

    # Step 1: Connect to Milvus
    connect_to_milvus()

    # Step 2: Check and create collection in Milvus
    check_and_create_collection(collection_name, dim)

    index_params = {
    "index_type": "IVF_FLAT",  # Example index type
    "metric_type": "L2",       # Example metric type, choose based on your embedding generation
    "params": {"nlist": 100}   # Example parameter, adjust based on your requirements
    }

    create_index_for_collection(collection_name, "embedding", index_params)

    # Step 3: Extract text from PDF
    extracted_text = extract_pdf_text(pdf_path)
    # Assuming a single block of text for simplicity, split or process as needed

    # Step 4: Generate embeddings (placeholder)
    embeddings = [generate_text_embeddings(extracted_text)]  # Modify as needed for real embeddings generation

    # Step 5: Insert embeddings into Milvus
    text_ids = [1]  # Unique IDs for each text segment, generate or modify as needed
    insert_embeddings(collection_name, text_ids, embeddings)

    # After inserting embeddings and before searching
    load_collection_into_memory(collection_name)


    # Optional Step 6: Demonstrate search (with a query of 'justice')
    query_embedding = generate_text_embeddings(query)
    search_results = search_embeddings(collection_name, [query_embedding], top_k=3)
    print(f"Search Results for {query}:", search_results)

    # Disconnect from Milvus
    disconnect_milvus()
