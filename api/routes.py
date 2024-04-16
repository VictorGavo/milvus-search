from flask import Blueprint, request, jsonify, current_app
import os
from services.pdf_processing import generate_text_embeddings, get_document_text_by_id, insert_document_text, process_pdf, store_data_in_systems
from util.milvus_operations import (
    connect_to_milvus, 
    check_and_create_collection,
    initialize_milvus_system,
    milvus_insert, 
    search_embeddings, 
    load_collection_into_memory,
)
from pymilvus import Collection

def create_api_blueprint():
    api_blueprint = Blueprint('api', __name__)

    @api_blueprint.route('/create_collection', methods=['POST'])
    def create_collection():
        data = request.json
        collection_name = data.get('collection_name')
        dim = data.get('dim', 1536)  # Default dimension
        initialize_milvus_system(collection_name, dim)
        return jsonify({"message": f"Collection {collection_name} initialized successfully."}), 200


    @api_blueprint.route('/populate_collection', methods=['POST'])
    def populate_collection():
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
        query_data = request.json
        query_text = query_data.get('query', '')
        collection_name = "arcadia_test"
        query_embedding = generate_text_embeddings(query_text)
        load_collection_into_memory(collection_name)
        search_results = search_embeddings(collection_name, [query_embedding], top_k=3)

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

        return jsonify({"results": detailed_results}), 200


    @api_blueprint.route('/discuss', methods=['POST'])
    def discuss_query():
        # This could be used for further interaction with the query results,
        # such as additional data processing, user feedback, or other analysis.
        query_result_id = request.json.get('result_id')
        discussion_text = request.json.get('text')
        # Implement functionality to handle the discussion/analysis
        return jsonify({"message": "Discussion recorded.", "result_id": query_result_id, "text": discussion_text})

    return api_blueprint

