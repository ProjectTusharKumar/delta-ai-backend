import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from server import (
    get_database,
    extract_query_filters,
    extract_context_and_schema_name,
    tokenizer,
    model,
    embedding_model,
    IGNORED_WORDS,
    NON_PERSON_WORDS
)
import numpy as np
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

def vector_search_employees(query, db, top_k=5):
    employees = list(db['employees'].find(
        {},
        {"_id": 0, "embedding": 1, "name": 1, "skills": 1, "currently on": 1, "projects": 1}
    ))

    valid_employees = []
    valid_embeddings = []
    for emp in employees:
        emb = emp.get('embedding')
        if isinstance(emb, list) and len(emb) == 384 and all(isinstance(x, (float, int)) for x in emb):
            valid_employees.append(emp)
            valid_embeddings.append(emb)

    if not valid_embeddings:
        return []

    # Convert to NumPy array and normalize
    embeddings = np.array(valid_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)

    # Encode query as a clean 1-D NumPy array
    query_emb = embedding_model.encode(query, convert_to_numpy=True)
    query_vec = np.asarray(query_emb, dtype=np.float32).flatten()
    q_norm = np.linalg.norm(query_vec)
    if q_norm < 1e-10:
        return []
    norm_query = query_vec / (q_norm + 1e-10)

    if embeddings.shape[1] != query_vec.shape[0]:
        # Dimension mismatch, return empty or log error
        return []

    # Use broadcasting for cosine similarity
    similarities = np.dot(norm_embeddings, norm_query.T).flatten()

    # Pick top-k
    top_idx = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        emp = valid_employees[idx].copy()
        emp['similarity'] = float(similarities[idx])
        results.append(emp)

    return results

@app.route("/api/chat", methods=["POST"])
def chat():
    queries = request.json.get('queries', []) or []
    results = {}
    db = get_database()
    coll = db['employees']

    for i, q in enumerate(queries, 1):
        if 'who' in q.lower():
            # Extract quoted phrases and treat them as NON_PERSON_WORDS for this query
            quoted_phrases = re.findall(r'\(([^)]+)\)', q)
            dynamic_non_person_words = set(NON_PERSON_WORDS)
            dynamic_non_person_words.update([phrase.lower() for phrase in quoted_phrases])
            # 1. Extract schema_name (fields) and person name from the query
            context_and_schema = extract_context_and_schema_name(q)
            schema_names = context_and_schema.get("schema_name", [])
            # Remove dynamic NON_PERSON_WORDS from detected person_names (case-insensitive, ignore leading/trailing spaces and brackets)
            def clean_word(word):
                return word.lower().strip().strip('()')
            cleaned_dynamic_non_person_words = set(clean_word(npw) for npw in dynamic_non_person_words)
            # Robust: Remove any person name that is fully or partially inside any parenthesized phrase
            parenthesized_tokens = set()
            for phrase in quoted_phrases:
                for token in phrase.split():
                    parenthesized_tokens.add(token.lower().strip('()'))
            person_names = [pn for pn in context_and_schema.get("context", [])
                            if clean_word(pn) not in cleaned_dynamic_non_person_words
                            and clean_word(pn) not in parenthesized_tokens]
            # 2. Remove ignored words for search text
            words = [w for w in q.split() if w.lower() not in IGNORED_WORDS]
            # 3. Remove schema words and dynamic NON_PERSON_WORDS from words list
            for schema in schema_names:
                schema_words = schema.split()
                words = [w for w in words if w.lower() not in [sw.lower() for sw in schema_words]]
            words = [w for w in words if w.lower() not in [npw.lower() for npw in dynamic_non_person_words]]
            # 4. The remaining words/phrase is the search value
            search_value = ' '.join(words).strip()
            db_employees = list(db['employees'].find({}, {"_id": 0}))
            vector_results = []
            if schema_names and search_value:
                for field in schema_names:
                    # Embed the search value
                    query_emb = embedding_model.encode(search_value, convert_to_numpy=True)
                    valid_employees = []
                    valid_embeddings = []
                    for emp in db_employees:
                        emb = emp.get(f'{field}_embedding')
                        if isinstance(emb, list):
                            # Optional: If person name is present, filter by name
                            if person_names:
                                emp_name = emp.get('name', '').lower()
                                if not any(pn.lower() in emp_name for pn in person_names):
                                    continue
                            # Remove embedding fields from result for clarity
                            emp_clean = {k: v for k, v in emp.items() if not k.endswith('_embedding')}
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
                    # Only get the top 1 result for clarity
                    if len(similarities) > 0:
                        idx = int(np.argmax(similarities))
                        emp = valid_employees[idx].copy()
                        emp['similarity'] = float(similarities[idx])
                        emp['vector_field'] = field
                        emp['vector_query_value'] = search_value
                        vector_results.append(emp)
            results[f'query{i}'] = {
                'query': q,
                'vector_search_value': search_value,
                'vector_result': vector_results[0] if vector_results else {},
                'schema_names': schema_names,
                'person_names': person_names,
                'info': 'Top field-specific vector search result with optional person name filtering.'
            }
            continue

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
            results[f'query{i}'] = {'query': q, 'mongo_query': filters, 'result': result}
        else:
            context_and_schema = extract_context_and_schema_name(q)
            schema_names = context_and_schema.get("schema_name", [])

            if schema_names:
                for field in schema_names:
                    match = re.search(rf"{field}.*?(\w+)", q, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        mongo_query = {field: {"$regex": value, "$options": "i"}}
                        result = list(coll.find(mongo_query, {"_id": 0}))
                        results[f'query{i}'] = {
                            'query': q,
                            'mongo_query': mongo_query,
                            'result': result,
                            'info': 'Used regex search for free-form query'
                        }
                        break
                else:
                    # Fallback to T5 if regex fails
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
                        results[f'query{i}'] = {
                            'query': q,
                            'mongo_query': mongo_query,
                            'result': result,
                            'generated_query_str': generated_query_str
                        }
                    except Exception as e:
                        results[f'query{i}'] = {**context_and_schema, 'error': f"T5 model failed: {e}"}
            else:
                # Full T5 fallback
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
                    results[f'query{i}'] = {
                        'query': q,
                        'mongo_query': mongo_query,
                        'result': result,
                        'generated_query_str': generated_query_str
                    }
                except Exception as e:
                    results[f'query{i}'] = {**context_and_schema, 'error': f"T5 model failed: {e}"}

    return jsonify(results)

@app.route("/api/field_vector_search", methods=["POST"])
def field_vector_search():
    data = request.json
    field = data.get("field")  # e.g., "skills"
    query = data.get("query")  # e.g., "Python and Data Science"
    limit = int(data.get("limit", 5))
    if not field or not query:
        return jsonify({"error": "'field' and 'query' are required"}), 400

    db = get_database()
    collection = db["employees"]
    embedding_field = f"{field}_embedding"
    index_name = f"{field}_vector_index"  # Make sure this index exists in Atlas

    # Generate embedding for the query
    query_embedding = embedding_model.encode(query).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": embedding_field,
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {
            "$project": {
                "name": 1,
                field: 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
