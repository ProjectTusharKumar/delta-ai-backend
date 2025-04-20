import os
import ssl
import re
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# Load embedding model
print("Loading embedding model (all-mpnet-base-v2)...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
print("Embedding model loaded.")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# MongoDB configuration

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://itstusharkumar15:admin@cluster0.wnyhv.mongodb.net/mydatabase?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true"
)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "user_db")

# Mongo client and database
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=20000
)
db = client[MONGO_DB_NAME]

# Collection name for employees data and embeddings
collection_name = 'employees'
employees_collection = db[collection_name]

# Ensure Vector Search index exists on 'docEmbedding'
def ensure_vector_index():
    idx = SearchIndexModel(
        name="employees_vector_idx",
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "docEmbedding": {
                        "type": "vectorSearch",
                        "dimensions": embedding_model.get_sentence_embedding_dimension(),
                        "similarity": "cosine"
                    }
                }
            }
        }
    )
    try:
        employees_collection.create_search_index(idx)
        print("Vector search index ensured.")
    except Exception as e:
        logging.error(f"Failed to create vector index: {e}")

# Call once on startup to ensure index
ensure_vector_index()

# Helper: build a single text string from record fields
def build_text_blob(rec: dict) -> str:
    parts = []
    for k, v in rec.items():
        if k in ['_id'] or v is None:
            continue
        parts.append(f"{k}: {v}")
    return "; ".join(parts)

# API: Upload .xlsx, store records with docEmbedding
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Reading Excel error: {e}"}), 500

    records = df.to_dict(orient="records")
    for rec in records:
        text = build_text_blob(rec)
        emb = embedding_model.encode(text).tolist()
        rec['docEmbedding'] = emb

    try:
        collection = db[collection_name]
        collection.insert_many(records)
    except Exception as e:
        return jsonify({"status": "error", "message": f"DB insert error: {e}"}), 500

    return jsonify({"status": "success", "message": f"Uploaded {len(records)} records with embeddings."})

# API: Semantic chat search via vectorSearch
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    queries = data.get('queries', []) or []
    k = data.get('k', 5)
    results = {}
    for idx, q in enumerate(queries, start=1):
        # Compute query embedding
        q_emb = embedding_model.encode(q).tolist()
        # Build aggregation pipeline
        stage = {
            "$vectorSearch": {
                "path": "docEmbedding",
                "queryVector": q_emb,
                "k": k
            }
        }
        pipeline = [
            stage,
            {"$project": {"score": {"$meta": "searchScore"}, "_id": 0}}
        ]
        try:
            collection = db[collection_name]
            docs = list(collection.aggregate(pipeline))
        except Exception as e:
            logging.error(f"Vector search error for query '{q}': {e}")
            docs = []
        results[f"query{idx}"] = {"query": q, "results": docs}

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host="0.0.0.0", port=port, debug=True)
