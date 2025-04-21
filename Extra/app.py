import os
import requests
from flask import Flask, request, jsonify
from pymongo import MongoClient
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from server import SPELLING_CORRECTIONS, schema_name, SPECIAL_schema_name

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MONGO_URI = os.getenv("mongodb+srv://itstusharkumar15:admin@cluster0.wnyhv.mongodb.net/mydatabase?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "user_db")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_TOKEN,
)

app = Flask(__name__)

model_name = "distilgpt2"  # Use a small, resource-friendly model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def hf_zephyr_chat(prompt, model="HuggingFaceH4/zephyr-7b-beta", max_tokens=512):
    completion = hf_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return completion.choices[0].message['content'] if hasattr(completion.choices[0], 'message') else str(completion)

def generate_intent(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def extract_intent_llm(query):
    # Use spaCy for name extraction
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Use spelling correction and schema matching
    words = query.lower().split()
    # Apply spelling correction
    corrected_words = [SPELLING_CORRECTIONS.get(w, w) for w in words]
    corrected_query = " ".join(corrected_words)
    # Try to match schema_name and SPECIAL_schema_name
    found_schema = []
    for phrase, mapped_keyword in SPECIAL_schema_name.items():
        if phrase in corrected_query:
            found_schema.append(mapped_keyword)
    for word in corrected_words:
        for schema in schema_name:
            if schema in word:
                found_schema.append(schema)
    # Remove duplicates
    found_schema = list(set(found_schema))
    # Compose intent dict
    intent = {}
    if found_schema:
        intent["field"] = found_schema[0]
    if persons:
        intent["person"] = persons[0]
    return intent

def hf_minilm_embed(text):
    api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result.get("embedding", [])

def get_database():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]

@app.route("/api/chat", methods=["POST"])
def chat():
    queries = request.json.get('queries', []) or []
    results = {}
    db = get_database()
    coll = db['employees']
    for i, q in enumerate(queries, 1):
        # 1. Use LLM to extract intent
        intent = extract_intent_llm(q)
        field = intent.get("field")
        person = intent.get("person")
        action = intent.get("action")
        # 2. If both field and person, do direct lookup
        if field and person:
            mongo_query = {"name": person}
            projection = {"_id": 0, field: 1, "name": 1}
            result = list(coll.find(mongo_query, projection))
            # Remove all fields ending with '_embedding' from each result
            result = [
                {k: v for k, v in doc.items() if not re.search(r'_embedding$', k)}
                for doc in result
            ]
            results[f'query{i}'] = {
                "query": q,
                "mongo_query": mongo_query,
                "projection": projection,
                "result": result,
                "info": "LLM intent extraction and direct lookup."
            }
            continue
        # 3. If only field, do vector search on that field
        elif field:
            search_value = intent.get("value") or intent.get("field_value") or q
            db_employees = list(coll.find({}, {"_id": 0}))
            vector_results = []
            query_emb = hf_minilm_embed(search_value)
            valid_employees = []
            valid_embeddings = []
            for emp in db_employees:
                emb = emp.get(f'{field}_embedding')
                if isinstance(emb, list):
                    emp_clean = {k: v for k, v in emp.items() if not re.search(r'_embedding$', k)}
                    valid_employees.append(emp_clean)
                    valid_embeddings.append(emb)
            if valid_embeddings:
                import numpy as np
                embeddings = np.array(valid_embeddings, dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norm_embeddings = embeddings / (norms + 1e-10)
                query_vec = np.asarray(query_emb, dtype=np.float32).flatten()
                q_norm = np.linalg.norm(query_vec)
                if q_norm >= 1e-10 and embeddings.shape[1] == query_vec.shape[0]:
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
                'schema_names': [field],
                'vector_search_value': search_value,
                'name': vector_results[0]['name'] if vector_results and 'name' in vector_results[0] else None
            }
            continue
        # 4. Fallback: error
        else:
            results[f'query{i}'] = {
                "query": q,
                "error": "Could not extract intent or person from query."
            }
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)