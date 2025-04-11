import os
import ssl
import re
import json
import logging
import spacy
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from rapidfuzz import fuzz, process
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

# Configure logging (set debug level and format)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
print("Logging is configured at DEBUG level.")

# Load spaCy model for text processing
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded: en_core_web_sm")

# Initialize Flask app and enable CORS (adjust for production)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
print("Flask app initialized and CORS enabled.")

# MongoDB configuration
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://itstusharkumar15:admin@cluster0.wnyhv.mongodb.net/mydatabase?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true"
)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "user_db")
print(f"MongoDB URI: {MONGO_URI}")
print(f"MongoDB Database Name: {MONGO_DB_NAME}")

def get_mongo_client():
    print("Attempting to connect to MongoDB...")
    logging.debug(f"Connecting to MongoDB using URI: {MONGO_URI}")
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,  # For development/testing only.
        serverSelectionTimeoutMS=20000
    )
    print("MongoDB client created.")
    return client

def get_database():
    client = get_mongo_client()
    try:
        db = client.get_default_database()
        print("Default database obtained.")
        logging.debug("Default database obtained from client.")
    except Exception as e:
        print("Default database not found, falling back to MONGO_DB_NAME.")
        logging.error(f"Error obtaining default database: {e}")
        db = client[MONGO_DB_NAME]
    return db

def check_db_connection():
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        logging.debug("MongoDB connection successful.")
        print("MongoDB connection successful.")
        return True, "MongoDB connection successful."
    except Exception as e:
        logging.error(f"MongoDB connection failed: {str(e)}")
        print(f"MongoDB connection failed: {str(e)}")
        return False, f"MongoDB connection failed: {str(e)}"

# -------------------- Helper Functions for Text Processing --------------------
SPELLING_CORRECTIONS = {
    "salry": "salary",
    "attndance": "attendance",
    "projetcs": "projects",
    "pastproject": "past projects",
    "past projetc": "past projects",
    "completed prject": "completed projects",
    "lst year project": "last year projects",
    "dateofjoining": "doj"
}

def correct_spelling(word):
    corrected = SPELLING_CORRECTIONS.get(word.lower(), word)
    logging.debug(f"Correcting '{word}' to '{corrected}'")
    print(f"Correcting '{word}' to '{corrected}'")
    return corrected

IGNORED_WORDS = {"both", "me", "can", "is", "and", "the", "for", "to", "of", "on", "please", ",", "retrieve", "fetch", "tell", "show", "whats", "summarize"}
NON_PERSON_WORDS = {"phone", "dob", "date", "number", "details", "projects", "salary", "attendance", "skills", "history"}

def extract_names(query):
    print(f"Extracting names from query: {query}")
    query = re.sub(r"(\w+)'s", r"\1", query.strip())
    words = [w for w in query.split() if w.lower() not in IGNORED_WORDS]
    cleaned_query = " ".join(words)
    doc = nlp(cleaned_query)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not persons:
        persons = [w for w in words if w.istitle() and w.lower() not in NON_PERSON_WORDS]
    logging.debug(f"Extracted names: {persons}")
    print(f"Extracted names: {persons}")
    return list(set(persons)) if persons else None

schema_name = [
    "dob", "doj", "salary", "phone number", "skills",
    "attendance", "last year projects", "past projects", "completed projects",
    "currently on", "total projects"
]

SPECIAL_schema_name = {
    "date of birth": "dob",
    "dob": "dob",
    "phone": "phone number",
    "phone no": "phone number",
    "work history": "total projects",
    "on going": "currently on",
    "project status": "currently on",
    "currently working on": "currently on",
    "working on recently": "currently on",
    "ongoing ones": "currently on",
    "ongoing projects": "currently on",
    "join date": "doj",
    "hired": "doj",
    "earning": "salary",
    "paid ": "salary"
}

def find_best_match(query, query_words):
    print(f"Finding best match in query: {query}")
    found_schema = []
    query_lower = query.lower()
    for phrase, mapped_keyword in SPECIAL_schema_name.items():
        if phrase in query_lower:
            found_schema.append(mapped_keyword)
            print(f"Matched special phrase '{phrase}' to '{mapped_keyword}'")
    for word in map(correct_spelling, query_words):
        result = process.extractOne(word, schema_name, scorer=fuzz.partial_ratio)
        if result:
            match, score, _ = result  # Unpack and ignore the index
            if score > 80 and match in query_lower:
                found_schema.append(match)
                print(f"Fuzzy matched word '{word}' to '{match}' with score {score}")
    logging.debug(f"Matched schema names: {found_schema}")
    print(f"Matched schema names: {found_schema}")
    return list(set(found_schema))

def generate_mongo_query_via_ai(employee_name, requested_fields):
    projection = {field: 1 for field in requested_fields}
    projection["_id"] = 0
    mongo_query = {
        "database": "mydatabase",
        "find": "employees",
        "projection": projection,
        "query": {
            "name": employee_name
        }
    }
    logging.debug(f"Generated Mongo query: {mongo_query}")
    print(f"Generated Mongo query: {mongo_query}")
    return mongo_query

def get_employee_data(employee_name, requested_fields):
    print(f"Getting data for employee '{employee_name}' with requested fields: {requested_fields}")
    mongo_query = generate_mongo_query_via_ai(employee_name, requested_fields)
    try:
        db = get_database()
        collection = db["employees"]
        employee_exists = collection.find_one({"name": employee_name})
        if not employee_exists:
            error_msg = f"No employee found with name {employee_name}"
            print(error_msg)
            logging.error(error_msg)
            return {"error": error_msg}
        projected_data = collection.find_one({"name": employee_name}, mongo_query["projection"])
        if not projected_data:
            error_msg = f"No requested fields found for {employee_name}"
            print(error_msg)
            logging.error(error_msg)
            return {"error": error_msg}
        filtered_data = {k: v for k, v in projected_data.items() if v is not None}
        if not filtered_data:
            error_msg = f"None of the requested fields were found for {employee_name}"
            print(error_msg)
            logging.error(error_msg)
            return {"error": error_msg}
        logging.debug(f"Employee data found: {filtered_data}")
        print(f"Employee data found: {filtered_data}")
        return {"mongo_query": mongo_query, "data": filtered_data}
    except Exception as e:
        error_msg = f"MongoDB query failed: {mongo_query}. Error: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return {"error": f"Failed to fetch data for {employee_name} from MongoDB."}

def extract_context_and_schema_name(query):
    print(f"Extracting context and schema from query: {query}")
    query = query.strip()
    for wrong, correct in SPELLING_CORRECTIONS.items():
        query = query.replace(wrong, correct)
        print(f"Corrected spelling: '{wrong}' -> '{correct}'")
    context = extract_names(query)
    query_words = query.split()
    found_schema = find_best_match(query, query_words)
    response = {"query": query, "context": context, "schema_name": found_schema}
    if context:
        employee_result = get_employee_data(context[0], [s.lower() for s in found_schema])
        response["employee_data"] = employee_result
        print(f"Employee result: {employee_result}")
    else:
        error_msg = "No valid employee name found in query."
        response["error"] = error_msg
        print(error_msg)
    logging.debug(f"Extracted context and schema: {response}")
    return response

# -------------------- API Endpoints --------------------

@app.route('/register', methods=['POST'])
def register():
    print("Processing registration request.")
    data = request.json
    username = data.get("username")
    password = data.get("password")  # Plain text (not secure for production)
    email = data.get("email")
    if not username or not password or not email:
        error_msg = "All fields are required"
        print(error_msg)
        return jsonify({"error": error_msg}), 400
    db = get_database()
    users_collection = db["users"]
    if users_collection.find_one({"username": username}):
        error_msg = "Username already exists"
        print(error_msg)
        return jsonify({"error": error_msg}), 400
    users_collection.insert_one({"username": username, "password": password, "email": email})
    logging.info(f"User {username} registered successfully.")
    print(f"User {username} registered successfully.")
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    print("Processing login request.")
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        error_msg = "Username and password required"
        print(error_msg)
        return jsonify({"message": error_msg}), 400
    db = get_database()
    users_collection = db["users"]
    user = users_collection.find_one({"username": username})
    if user and user["password"] == password:
        logging.info("Login successful.")
        print("Login successful.")
        return jsonify({
            "message": "Login successful",
            "email": user.get("email"),
            "username": user.get("username")
        }), 200
    else:
        error_msg = "Invalid username or password"
        print(error_msg)
        return jsonify({"message": error_msg}), 401

@app.route("/api/check_connection", methods=["GET"])
def api_check_connection():
    print("Checking MongoDB connection...")
    connected, message = check_db_connection()
    if connected:
        return jsonify({"connection": True, "message": message})
    else:
        return jsonify({"connection": False, "message": message}), 500

# -------------------- Updated Employee Save Endpoint --------------------
@app.route("/save-employee", methods=["POST"])
def save_employee():
    print("Processing employee save request.")
    data = request.get_json()
    # Required fields (adjust as needed)
    required_fields = ["name", "dob", "phone", "email", "skills", "doj", "salary", "feedback"]
    for field in required_fields:
        if not data.get(field):
            error_msg = f"{field} is required"
            print(error_msg)
            return jsonify({"error": error_msg}), 400

    # Split data into employee info and feedback
    employee_info = {
        "name": data.get("name"),
        "dob": data.get("dob"),
        "phone": data.get("phone"),
        "email": data.get("email"),
        "skills": data.get("skills"),
        "doj": data.get("doj"),
        "salary": data.get("salary"),
        "customFields": data.get("customFields", [])
    }
    feedback_value = data.get("feedback")
    print(f"Employee info: {employee_info}")
    print(f"Feedback: {feedback_value}")

    db = get_database()
    employees_collection = db["employees"]
    feedbacks_collection = db["feedbacks"]

    try:
        # Insert employee information first
        employee_result = employees_collection.insert_one(employee_info)
        employee_id = employee_result.inserted_id
        print(f"Inserted employee with ID: {employee_id}")

        # Insert feedback in a separate collection with reference to employee_id
        feedback_doc = {
            "employee_id": str(employee_id),
            "feedback": feedback_value
        }
        feedbacks_collection.insert_one(feedback_doc)
        print("Feedback saved successfully.")

        return jsonify({"message": "Employee and feedback saved successfully!"}), 201
    except Exception as e:
        error_msg = f"Error saving employee: {e}"
        logging.error(error_msg)
        print(error_msg)
        return jsonify({"error": "Failed to save employee data"}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    print("Processing file upload request.")
    if 'file' not in request.files:
        error_msg = "No file part"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 400
    file = request.files['file']
    table_name = request.form.get('table_name')
    if file.filename == '':
        error_msg = "No selected file"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 400
    if not table_name:
        error_msg = "Table name is required"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 400
    try:
        df = pd.read_excel(file)
        df.columns = [col.lower() for col in df.columns]
        logging.debug(f"DataFrame columns after lowercasing: {df.columns.tolist()}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        records = df.to_dict(orient="records")
        db = get_database()
        collection = db[table_name]
        collection.insert_many(records)
        print(f"Data uploaded to collection '{table_name}' successfully!")
        return jsonify({"status": "success", "message": f"Data uploaded to collection '{table_name}' successfully!"})
    except Exception as e:
        error_msg = f"Failed to upload data: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    print("Processing chat request.")
    data = request.json
    queries = data.get("queries", [])
    results = {}
    for i, query in enumerate(queries, start=1):
        print(f"Processing query {i}: {query}")
        result = extract_context_and_schema_name(query)
        results[f"query{i}"] = result
    logging.debug(f"Chat results: {results}")
    print(f"Chat results: {results}")
    return jsonify(results)

@app.route("/api/collections", methods=["GET"])
def get_all_collections():
    print("Fetching all collection names.")
    try:
        db = get_database()
        collections = db.list_collection_names()
        logging.debug(f"Collections: {collections}")
        print(f"Collections: {collections}")
        return jsonify({"collections": collections})
    except Exception as e:
        error_msg = f"Error fetching collection names: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route("/api/collection", methods=["GET"])
def get_collection_data():
    collection_name = request.args.get("name")
    print(f"Fetching data from collection: {collection_name}")
    if not collection_name:
        error_msg = "Missing collection name"
        print(error_msg)
        return jsonify({"error": error_msg}), 400
    try:
        db = get_database()
        collection = db[collection_name]
        data = list(collection.find({}, {"_id": 0}))
        logging.debug(f"Data from collection '{collection_name}': {data}")
        print(f"Data from collection '{collection_name}': {data}")
        return jsonify({"collection_name": collection_name, "data": data})
    except Exception as e:
        error_msg = f"Failed to fetch data from collection {collection_name}: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route("/api/employees", methods=["GET"])
def get_employees():
    print("Fetching employee data.")
    try:
        db = get_database()
        collection = db["employees"]
        data = list(collection.find({}, {"_id": 0}))
        logging.debug(f"Employees: {data}")
        print(f"Employees: {data}")
        return jsonify({"employees": data})
    except Exception as e:
        error_msg = f"Failed to fetch employees: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
