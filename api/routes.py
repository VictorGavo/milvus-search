from flask import Blueprint, request, jsonify, current_app, session
import os
from services.openai_services import call_gpt_for_conversation, call_gpt_for_summary, generate_text_embeddings
from services.pdf_processing import get_document_text_by_id, process_pdf, store_data_in_systems
from util.milvus_operations import (
    initialize_milvus_system,
    search_embeddings, 
    load_collection_into_memory,
)

def create_api_blueprint():
    api_blueprint = Blueprint('api', __name__)

    @api_blueprint.route('/create_collection', methods=['POST'])
    def create_collection():
        """
        Creates a new Milvus collection based on provided configuration via a POST request.

        Inputs:
        - JSON body containing:
        - 'collection_name' (str): Name for the new collection.
        - 'dim' (int, optional): Dimension for the embeddings, with a default of 1536.

        Processes:
        - Initializes Milvus system based on the provided 'collection_name' and 'dim'.
        - Ensures the collection is created or reports an error if it already exists.

        Outputs:
        - JSON response indicating success or failure of the operation.
        - HTTP status code 200 on success, appropriate error code on failure.
        """
        data = request.json
        collection_name = data.get('collection_name')
        dim = data.get('dim', 1536)  # Default dimension
        initialize_milvus_system(collection_name, dim)
        return jsonify({"message": f"Collection {collection_name} initialized successfully."}), 200


    @api_blueprint.route('/populate_collection', methods=['POST'])
    def populate_collection():
        """
        Handles the upload and processing of PDF files to populate the Milvus collection with text embeddings.

        Inputs:
        - Multi-part form data with one or more PDF files.

        Processes:
        - Validates presence and type of uploaded documents.
        - For each valid document, extracts text, generates embeddings, and stores both in respective databases.
        - Reports errors for any invalid files or processing issues.

        Outputs:
        - JSON response listing the outcome for each file processed (either 'saved' or error message).
        - HTTP status code 200 if all files processed successfully, 400 or 500 if there are errors.
        """
        collection_name = current_app.config.get('MILVUS_COLLECTION')
        if not collection_name:
            return jsonify({'error': 'Milvus collection name is not set'}), 500
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        if not upload_folder:
            return jsonify({'error': 'Upload folder configuration is missing or incorrect'}), 500

        print("Upload folder is set to:", upload_folder)

        if 'document' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('document')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        results = []
        for file in files:
            if file.filename == '':
                results.append({'file': 'empty', 'status': 'skipped'})
                continue
            filename = allowed_file(file.filename)
            if not filename:
                results.append({'file': file.filename, 'status': 'file type not allowed'})
                continue

            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Process the PDF
            vector_db, text_db = process_pdf(file_path)
            store_data_in_systems(current_app.config.get('MILVUS_COLLECTION'), vector_db, text_db)
            results.append({'file': filename, 'status': 'saved', 'path': file_path})

        return jsonify({'message': 'PDF(s) processed and data stored successfully.', 'results': results}), 200


    def allowed_file(filename):
        ALLOWED_EXTENSIONS = {'pdf'}
        if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            return filename  # Return the filename if it's allowed
        return None  # Return None if the file is not allowed

    @api_blueprint.route('/query', methods=['POST'])
    def query_document():
        """
        Processes a textual query against stored embeddings in the Milvus database and retrieves relevant documents.

        Inputs:
        - JSON body containing:
        - 'query' (str): Text query input by the user.

        Processes:
        - Generates embeddings for the query text.
        - Searches the Milvus collection for the most relevant documents based on the query embeddings.
        - Compiles detailed results including document texts and metadata from the SQL database.

        Outputs:
        - JSON response containing detailed results and a summary of the found documents.
        - HTTP status code 200 on success, 500 if there is an error during processing.
        """
        query_data = request.json
        query_text = query_data.get('query', '')
        collection_name = "arcadia_test"

        # Generate embeddings and search
        query_embedding = generate_text_embeddings(query_text)
        load_collection_into_memory(collection_name)
        search_results = search_embeddings(collection_name, [query_embedding], top_k=3)

        # Prepare text for summary
        documents_for_summary = " ".join([get_document_text_by_id(result['text_id'])['text'] for result in search_results])
        summary = call_gpt_for_summary(documents_for_summary)

        # Store results and summary in session for later use in discussion
        session['query_results'] = search_results
        session['summary'] = summary

        detailed_results = []
        for result in search_results:
            document_details = get_document_text_by_id(result['text_id'])
            if document_details:
                result_detail = {
                    "id": result['id'],
                    "score": result['score'],
                    "text_id": result['text_id'],
                    "text": document_details['text'],
                    "document_name": document_details['document_name'],
                    "page_number": document_details['page_number']
                }
                detailed_results.append(result_detail)
            else:
                print(f"Document details not found for text ID: {result['text_id']}")

        return jsonify({"results": detailed_results, "summary": summary}), 200


    @api_blueprint.route('/discuss', methods=['POST'])
    def discuss_query():
        """
        Engages a conversational model to discuss query results based on a user's follow-up question.

        Inputs:
        - JSON body containing:
        - 'question' (str): User's follow-up question for discussion.

        Processes:
        - Retrieves the context of previous query results and user question from session storage.
        - Utilizes a conversational AI model to generate a response based on the combined context.

        Outputs:
        - JSON response with the AI-generated answer to the user's question.
        - HTTP status code 200 on successful interaction, 400 if session data is missing.
        """
        data = request.json
        user_question = data.get('question', '')

        if 'summary' not in session or 'query_results' not in session:
            return jsonify({"error": "No active session or previous query data found"}), 400

        # Prepare the context and the user question for the GPT model
        context = session['summary']  # This assumes that the summary is stored in the session
        message = user_question

        # Call the GPT model to discuss the query based on the provided context and message
        gpt_response = call_gpt_for_conversation(message, context)

        return jsonify({"response": gpt_response}), 200


    return api_blueprint

