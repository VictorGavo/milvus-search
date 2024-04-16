# Document Processing API

This project provides a Flask-based API for processing PDF documents, extracting text, generating embeddings, and storing them in both Milvus and a SQLite database. It allows querying these embeddings for matching text entries based on input queries.

## Features

- Process PDF documents to extract text and generate embeddings.
- Store embeddings in Milvus and text data in SQLite with associated metadata.
- Query the system with text to find relevant document entries.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8+
- pip
- virtualenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourprojectname.git
   cd yourprojectname
   ```

2. Set up a Python virtual environment and activate it:
   ```bash
   python -m virtualenv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```bash
   python manage.py init_db
   ```

### Usage

Run the Flask application:
```bash
flask run
```

This will start the server on `http://127.0.0.1:5000/`. The API endpoints can be accessed from this base URL.

### API Endpoints

- `POST /api/create_collection` - Create a new collection and index in Milvus.
- `POST /api/populate_collection` - Upload PDF documents and process them into the system.
- `POST /api/query` - Send a text query to retrieve relevant document pages.

### Testing

Explain how to run the automated tests for this system.

## Built With

- [Flask](https://flask.palletsprojects.com/) - The web framework used.
- [Milvus](https://milvus.io/) - Vector database for storing embeddings.
- [SQLite](https://www.sqlite.org/index.html) - SQL database engine for storing text data.

## Authors

- **Victor Gavojdea** - *Initial work* - [VictorGavo](https://github.com/VictorGavo)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- ai.douglas.life for their custom Code God Mode GPT

