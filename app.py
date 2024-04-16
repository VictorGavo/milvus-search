from flask import Flask
from api.routes import create_api_blueprint
from util.milvus_operations import connect_to_milvus
import os

app = Flask(__name__)

# Establish connection to Milvus
connect_to_milvus()

# Register the API blueprint
api_blueprint = create_api_blueprint()
app.register_blueprint(api_blueprint, url_prefix='/api')
app.config['MILVUS_COLLECTION'] = 'arcadia_test'

app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['MILVUS_COLLECTION'] = 'arcadia_test'  # Ensure this is set
    app.run(debug=True)
