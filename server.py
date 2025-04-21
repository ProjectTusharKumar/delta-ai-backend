import os
import ssl
import re
import json
import logging
import spacy
import pandas as pd
import numpy as np
import faiss
# import dateparser
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from rapidfuzz import fuzz, process
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
print("Logging is configured at DEBUG level.")

# Load spaCy model for text processing
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded: en_core_web_sm")

# Load transformer T5 model and tokenizer
print("Loading transformer model T5-base...")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
print("Transformer model T5-base loaded.")

# Load SentenceTransformer for embeddings
print("Loading embedding model (all-mpnet-base-v2")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
print("Embedding model loaded.")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://opulent-lamp-5g5qg7w7gqvwf7j96-3000.app.github.dev","https://dealta-ai.onrender.com"
]}})
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
    "dateofjoining": "doj",
    "employees": "employee"
}

IGNORED_WORDS = {"both", "me", "can", "is", "and", "the", "for", "to", "of", "on", "please", ",", "retrieve", "fetch", "tell", "show", "whats", "summarize", "who"}
NON_PERSON_WORDS = {"phone","Project Mu", "dob", "date", "number", "details", "projects", "salary", "attendance", "skills","Who", "history"}

def correct_spelling(word):
    corrected = SPELLING_CORRECTIONS.get(word.lower(), word)
    logging.debug(f"Correcting '{word}' to '{corrected}'")
    print(f"Correcting '{word}' to '{corrected}'")
    return corrected

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
    "completed": "completed projects",
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
            match, score, _ = result
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
    q = query.strip()
    filters = extract_query_filters(q)
    if filters:
        db = get_database()
        res = list(db['employees'].find(filters, {'_id':0}))
        context = [filters['name']] if 'name' in filters else []
        data = res[0] if len(res)==1 else res
        return {'query': q, 'context': context, 'schema_name': [],
                'employee_data': {'data': data, 'mongo_query': filters}}
    # fallback
    text = q
    for wrong, corr in SPELLING_CORRECTIONS.items():
        text = text.replace(wrong, corr)
    context = extract_names(text) or []
    words = text.split()
    fields = find_best_match(text, words)
    if context and fields:
        emp = get_employee_data(context[0], [f for f in fields])
    else:
        emp = {'error': 'Unable to parse query.'}
    return {'query': q, 'context': context, 'schema_name': fields, 'employee_data': emp}
    

# -------------------- New Helper: Extract Query Filters --------------------
def extract_query_filters(query):
    filters = {}
    query_lower = query.lower()

    # Handle "last month" queries for date of joining (doj)
    if "last month" in query_lower:
        today = datetime.today()
        first_day_this_month = today.replace(day=1)
        last_month_end = first_day_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        filters["doj"] = {
            "$gte": last_month_start.strftime("%Y-%m-%d"),
            "$lte": last_month_end.strftime("%Y-%m-%d")
        }

    # Numeric range filter for age (using updated regex)
    age_pattern = re.search(r'age\s+between\s+(\d+)\s*(?:and|to|-)\s*(\d+)', query_lower)
    if age_pattern:
        lower_age = int(age_pattern.group(1))
        upper_age = int(age_pattern.group(2))
        current_year = datetime.today().year
        dob_lower = f"{current_year - upper_age}-01-01"
        dob_upper = f"{current_year - lower_age}-12-31"
        filters["dob"] = {"$gte": dob_lower, "$lte": dob_upper}

    # Salary range filter (new addition)
    salary_pattern = re.search(r'salary\s+between\s+(\d+)\s*(?:and|to|-)\s*(\d+)', query_lower)
    if salary_pattern:
        lower_sal = int(salary_pattern.group(1))
        upper_sal = int(salary_pattern.group(2))
        filters["salary"] = {"$gte": lower_sal, "$lte": upper_sal}

    # Full details query e.g., "give details of John Doe"
    name_pattern = re.search(r'detail[s]?\s+(?:of|about)\s+([A-Za-z\s]+)', query, re.IGNORECASE)
    if name_pattern:
        filters["name"] = name_pattern.group(1).strip()

    return filters

   
# -------------------- Vector Embedding and FAISS Indexing --------------------
def compute_embedding(text):
    return embedding_model.encode(text)

def build_vector_index(records):
    # Build texts from records (concatenating selected fields)
    texts = [record.get("name", "") + " " + record.get("skills", "") for record in records]
    embeddings = np.array([compute_embedding(text) for text in texts]).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, texts

# -------------------- API Endpoints --------------------

@app.route('/register', methods=['POST'])
def register():
    print("Processing registration request.")
    data = request.json
    username = data.get("username")
    password = data.get("password")
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

@app.route("/save-employee", methods=["POST"])
def save_employee():
    print("Processing employee save request.")
    data = request.get_json()
    required_fields = ["name", "dob", "phone", "email", "skills", "doj", "salary", "feedback"]
    for field in required_fields:
        if not data.get(field):
            error_msg = f"{field} is required"
            print(error_msg)
            return jsonify({"error": error_msg}), 400

    employee_info = {
        "name": data.get("name"),
        "dob": data.get("dob"),
        "gender": data.get("gender"),
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
        employee_result = employees_collection.insert_one(employee_info)
        employee_id = employee_result.inserted_id
        print(f"Inserted employee with ID: {employee_id}")
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
    if file.filename == '':
        error_msg = "No selected file"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 400
    # Save all uploads to 'employees' collection
    collection_name = 'employees'
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as read_error:
        error_msg = f"Excel file reading error: {read_error}"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

    try:
        df.columns = [col.lower() for col in df.columns]
        records = df.to_dict(orient="records")
        for rec in records:
            for field in ["skills", "past projects", "currently on"]:
                value = rec.get(field)
                if isinstance(value, str) and value.strip():
                    rec[f"{field}_embedding"] = embedding_model.encode(value).tolist()
                    
    except Exception as conversion_error:
        error_msg = f"DataFrame conversion error: {conversion_error}"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

    try:
        db = get_database()
        collection = db[collection_name]
        collection.insert_many(records)
        # Build vector index for the uploaded records
        index, embeddings, texts = build_vector_index(records)
        print("Vector index built for uploaded records.")
        # (Optionally, store index details or persist embeddings as needed)
    except Exception as db_error:
        error_msg = f"Database insertion error: {db_error}"
        print(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

    success_msg = f"Data uploaded to collection '{collection_name}' successfully, and vector index created!"
    print(success_msg)
    return jsonify({"status": "success", "message": success_msg})

@app.route("/api/chat", methods=["POST"])
def chat():
    queries = request.json.get('queries', []) or []
    results = {}
    db = get_database()
    coll = db['employees']

    for i, q in enumerate(queries, 1):
        # Special case: handle 'List all employees' or similar queries
        if re.match(r'list all employees', q.strip(), re.IGNORECASE):
            result = list(coll.find({}, {'_id': 0, 'name': 1}))
            results[f'query{i}'] = {
                'query': q,
                'result': result,
                'info': 'Listed all employee names.'
            }
            continue

        if 'who' in q.lower():
            # --- Begin app.py vector search logic ---
            quoted_phrases = re.findall(r'\(([^)]+)\)', q)
            dynamic_non_person_words = set(NON_PERSON_WORDS)
            dynamic_non_person_words.update([phrase.lower() for phrase in quoted_phrases])
            context_and_schema = extract_context_and_schema_name(q)
            schema_names = context_and_schema.get("schema_name", [])
            def clean_word(word):
                return word.lower().strip().strip('()')
            cleaned_dynamic_non_person_words = set(clean_word(npw) for npw in dynamic_non_person_words)
            parenthesized_tokens = set()
            for phrase in quoted_phrases:
                for token in phrase.split():
                    parenthesized_tokens.add(token.lower().strip('()'))
            person_names = [pn for pn in context_and_schema.get("context", [])
                            if clean_word(pn) not in cleaned_dynamic_non_person_words
                            and clean_word(pn) not in parenthesized_tokens]
            words = [w for w in q.split() if w.lower() not in IGNORED_WORDS]
            for schema in schema_names:
                schema_words = schema.split()
                words = [w for w in words if w.lower() not in [sw.lower() for sw in schema_words]]
            words = [w for w in words if w.lower() not in [npw.lower() for npw in dynamic_non_person_words]]
            search_value = ' '.join(words).strip()
            db_employees = list(db['employees'].find({}, {"_id": 0}))
            vector_results = []
            if schema_names and search_value:
                for field in schema_names:
                    query_emb = embedding_model.encode(search_value, convert_to_numpy=True)
                    valid_employees = []
                    valid_embeddings = []
                    for emp in db_employees:
                        emb = emp.get(f'{field}_embedding')
                        if isinstance(emb, list):
                            if person_names:
                                emp_name = emp.get('name', '').lower()
                                if not any(pn.lower() in emp_name for pn in person_names):
                                    continue
                            # Remove all fields ending with '_embedding' from the result
                            emp_clean = {k: v for k, v in emp.items() if not re.search(r'_embedding$', k)}
                            valid_employees.append(emp_clean)
                            valid_embeddings.append(emb)
                    if not valid_embeddings:
                        continue
                    embeddings = np.array(valid_embeddings, dtype=np.float32)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norm_embeddings = embeddings / (norms + 1e-10)
                    query_vec = np.asarray(query_emb, dtype=np.float32).flatten()
                    q_norm = np.linalg.norm(query_vec)
                    if q_norm < 1e-10 or embeddings.shape[1] != query_vec.shape[0]:
                        continue
                    norm_query = query_vec / (q_norm + 1e-10)
                    similarities = np.dot(norm_embeddings, norm_query.T).flatten()
                    if len(similarities) > 0:
                        idx = int(np.argmax(similarities))
                        emp = valid_employees[idx].copy()
                        emp['similarity'] = float(similarities[idx])
                        emp['vector_field'] = field
                        emp['vector_query_value'] = search_value
                        vector_results.append(emp)
            results[f'query{i}'] = {
                'query': q,
                'schema_names': schema_names,
                'vector_search_value': search_value,
                'name': vector_results[0]['name'] if vector_results and 'name' in vector_results[0] else None
            }
            continue
        # --- End app.py vector search logic ---

        # --- Begin server.py rule-based and T5 fallback logic ---
        filters = extract_query_filters(q)
        if filters:
            if 'salary' in filters:
                projection = {'_id': 0, 'name': 1, 'salary': 1}
            elif 'dob' in filters and isinstance(filters['dob'], dict):
                projection = {'_id': 0, 'name': 1, 'doj': 1}
            elif 'doj' in filters and isinstance(filters['doj'], dict):
                projection = {'_id': 0, 'name': 1, 'doj': 1}
            elif 'name' in filters:
                projection = {'_id': 0}
            else:
                projection = {'_id': 0, 'name': 1}
            result = list(coll.find(filters, projection))
            # Remove all fields ending with '_embedding' from each result
            result = [
                {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                for doc in result
            ]
            results[f'query{i}'] = {'query': q, 'mongo_query': filters, 'result': result}
        else:
            context_and_schema = extract_context_and_schema_name(q)
            schema_names = context_and_schema.get("schema_name", [])
            context_names = context_and_schema.get("context", [])
            if schema_names and context_names:
                # If both a field (e.g. salary) and a person name (e.g. John Doe) are found, query by name and project the field
                field = schema_names[0]
                person = context_names[0]
                mongo_query = {"name": person}
                projection = {"_id": 0, field: 1, "name": 1}
                result = list(coll.find(mongo_query, projection))
                # Remove all fields ending with '_embedding' from each result
                result = [
                    {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                    for doc in result
                ]
                results[f'query{i}'] = {
                    'query': q,
                    'mongo_query': mongo_query,
                    'projection': projection,
                    'result': result,
                    'info': 'Field and person name extracted for direct lookup.'
                }
            elif schema_names:
                for field in schema_names:
                    match = re.search(rf"{field}.*?(\w+)", q, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        mongo_query = {field: {"$regex": value, "$options": "i"}}
                        result = list(coll.find(mongo_query, {"_id": 0}))
                        # Remove all fields ending with '_embedding' from each result
                        result = [
                            {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                            for doc in result
                        ]
                        results[f'query{i}'] = {
                            'query': q,
                            'mongo_query': mongo_query,
                            'result': result,
                            'info': 'Used regex search for free-form query'
                        }
                        break
                else:
                    prompt = (
                        f"Convert the following natural language request into a MongoDB query "
                        f"for the 'employees' collection: '{q}' Return only the JSON."
                    )
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    output_ids = model.generate(input_ids, max_length=150)
                    generated_query_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    try:
                        mongo_query = json.loads(generated_query_str)
                        result = list(coll.find(mongo_query, {"_id": 0}))
                        # Remove all fields ending with '_embedding' from each result
                        result = [
                            {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                            for doc in result
                        ]
                        results[f'query{i}'] = {
                            'query': q,
                            'mongo_query': mongo_query,
                            'result': result,
                            'generated_query_str': generated_query_str
                        }
                    except Exception as e:
                        results[f'query{i}'] = {**context_and_schema, 'error': f"T5 model failed: {e}"}
            else:
                prompt = (
                    f"Convert the following natural language request into a MongoDB query "
                    f"for the 'employees' collection: '{q}' Return only the JSON."
                )
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                output_ids = model.generate(input_ids, max_length=150)
                generated_query_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                try:
                    mongo_query = json.loads(generated_query_str)
                    result = list(coll.find(mongo_query, {"_id": 0}))
                    # Remove all fields ending with '_embedding' from each result
                    result = [
                        {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                        for doc in result
                    ]
                    results[f'query{i}'] = {
                        'query': q,
                        'mongo_query': mongo_query,
                        'result': result,
                        'generated_query_str': generated_query_str
                    }
                except Exception as e:
                    results[f'query{i}'] = {**context_and_schema, 'error': f"T5 model failed: {e}"}
        # --- End server.py rule-based and T5 fallback logic ---

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
        # Remove all fields ending with '_embedding' from each result
        data = [
            {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
            for doc in data
        ]
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