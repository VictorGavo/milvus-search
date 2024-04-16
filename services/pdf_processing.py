import fitz
import configparser
import hashlib
import os
import sqlite3
import hashlib
import datetime

from services.openai_services import generate_text_embeddings
from util.milvus_operations import initialize_milvus_system, search_embeddings, milvus_insert, load_collection_into_memory

config = configparser.ConfigParser()
config.read('config.ini')

def segment_pdf(pdf_path):
    """
    Segment the given PDF by page, extracting text for each page.

    Parameters:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - List of tuples (page_num, text) for each page.
    """

    doc = fitz.open(pdf_path)
    text_segments = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        text_segments.append((page_num, text))
    return text_segments

def generate_unique_id(pdf_path, page_num):
    """
    Generate a unique integer ID for each page using a hash of the PDF path and page number.

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - page_num (int): Page number.

    Returns:
    - A unique identifier as an integer.
    """
    # Create a hash of the PDF path and page number
    unique_string = f"{pdf_path}_{page_num}"
    hash_obj = hashlib.md5(unique_string.encode())
    doc_hash = hash_obj.hexdigest()
    
    # Use a portion of the hash to generate an integer
    unique_id = int(doc_hash[:8], 16)
    
    return unique_id

def generate_hash(text):
    """Generate SHA-256 hash of the text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_pdf(pdf_path):
    """
    Process the PDF to extract text by page, generate embeddings, and prepare metadata for storage.

    Parameters:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - tuple: Two dictionaries containing vector data and text data with metadata.
    """
    document_name = os.path.basename(pdf_path)
    segments = segment_pdf(pdf_path)
    vector_db = {}
    text_db = {}

    for page_num, text in segments:
        unique_id = generate_unique_id(pdf_path, page_num)
        embedding = generate_text_embeddings(text)
        text_hash = generate_hash(text)
        processed_at = datetime.datetime.now()

        # Store vector and text data with additional metadata
        vector_db[unique_id] = embedding
        text_db[unique_id] = {
            'DocumentName': document_name,
            'PageNumber': page_num,
            'Text': text,
            'TextHash': text_hash,
            'ProcessedAt': processed_at
        }

    return vector_db, text_db

def process_and_store_pdf(pdf_path, collection_name):
    """
    Full processing and storing pipeline for a given PDF.

    Parameters:
    - pdf_path (str): The path to the PDF file to process.
    - collection_name (str): The Milvus collection name where data will be stored.
    """
    # Initialize Milvus and prepare the collection
    initialize_milvus_system(collection_name)

    # Process the PDF and get data ready for storage
    vector_db, text_db = process_pdf(pdf_path)

    # Store data in Milvus and the SQL database
    store_data_in_systems(collection_name, vector_db, text_db)

def store_data_in_systems(collection_name, vector_db, text_db):
    """
    Stores embeddings in Milvus and corresponding text in an SQL database.

    Parameters:
    - collection_name (str): The name of the Milvus collection.
    - vector_db (dict): A dictionary of unique IDs to embeddings.
    - text_db (dict): A dictionary of unique IDs to text segments, each text segment should be a dictionary 
                      containing the document name, text, page number and optionally the processed time.
    """
    conn = sqlite3.connect('text.db')
    cursor = conn.cursor()

    for unique_id, embedding in vector_db.items():
        text_hash = text_db[unique_id]['TextHash']
        cursor.execute("SELECT COUNT(*) FROM document_texts WHERE TextHash = ?", (text_hash,))
        if cursor.fetchone()[0] == 0:
            milvus_insert(collection_name, [unique_id], [embedding])

            # Extract text details
            text_details = text_db[unique_id]
            document_name = text_details['DocumentName']
            text = text_details['Text']
            hash_text=text_details['TextHash']
            page_number = text_details['PageNumber']
            processed_at = text_details.get('ProcessedAt', None)

            # Insert text and related details into SQL database
            insert_document_text(unique_id, document_name, page_number, text, hash_text, processed_at)
            # Load vectorized data into milvus
            load_collection_into_memory(collection_name)
            print(f"Data stored for ID {unique_id}")
        else:
            print(f"Skipping insertion for ID {unique_id} as it already exists.")
    conn.commit()
    conn.close()


def insert_document_text(id, document_name, page_number, text, hash_text, processed_at=None):
    """
    Inserts document text details into the SQL database if the hash is unique.

    Parameters:
    - id (int): The unique identifier for the document text.
    - document_name (str): The name of the document.
    - page_number (int): The page number of the text within the document.
    - text (str): The actual text content of the document page.
    - hash_text (str): The hash of the text, used to check for duplicates.
    - processed_at (datetime): The timestamp when the document was processed; defaults to the current time if None.
    """
    # Use the current timestamp if no processing time is provided
    if not processed_at:
        processed_at = datetime.now()

    # Ensure ID is an integer
    id = int(id)

    # Connect to the SQLite database
    conn = sqlite3.connect('text.db')
    cursor = conn.cursor()

    # Check if a text with the same hash already exists
    cursor.execute("SELECT COUNT(*) FROM document_texts WHERE TextHash = ?", (hash_text,))
    if cursor.fetchone()[0] > 0:
        print(f"Document text with hash {hash_text} already exists. Skipping insert.")
        conn.close()  # Ensure the connection is closed before returning
        return
    
    try:
        cursor.execute("""
            INSERT INTO document_texts (ID, DocumentName, PageNumber, Text, TextHash, ProcessedAt)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id, document_name, page_number, text, hash_text, processed_at))
        conn.commit()
        print("Data inserted successfully for ID:", id)
    except sqlite3.IntegrityError as e:
        print(f"Failed to insert data into the database due to a unique constraint failure: {e}")
    except sqlite3.InterfaceError as e:
        print(f"Failed to insert data into the database: {e}")
    finally:
        conn.close()

def get_document_text_by_id(text_id):
    """
    Fetches document text along with document name and page number from the database by text ID.

    Parameters:
    - text_id (int): The ID of the text in the database.

    Returns:
    - dict: A dictionary containing text, document name, and page number if found, else None.
    """
    conn = sqlite3.connect('text.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Text, DocumentName, PageNumber FROM document_texts WHERE ID = ?", (text_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {'text': row[0], 'document_name': row[1], 'page_number': row[2]}
    return None


def search_documents(query_text):
    # Generate an embedding for the query text
    query_embedding = generate_text_embeddings(query_text)
    
    # Search Milvus for the top N similar embeddings
    milvus_results = search_embeddings(
        collection_name="arcadia_test",
        query_embeddings=[query_embedding],
        top_k=3
    )
    
    # Extract IDs from Milvus search results
    # Assuming each result in milvus_results has an attribute 'ids' that contains the matching IDs
    similar_ids = [int(id) for result in milvus_results for id in result.ids]
    
    # Retrieve the corresponding texts for these IDs from the SQLite database
    texts = [get_document_text_by_id(doc_id) for doc_id in similar_ids]
    
    return texts
