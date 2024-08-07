from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_elastic import RagElastic
import os
from uuid import uuid4

app = Flask(__name__)

CORS(app)

rag_elastic = RagElastic()

@app.route("/")
def index():
    return render_template('ui.html')

@app.post('/')
def handle_search():
    query = request.form.get('query', '')
    result = rag_elastic.make_search(query)
    return render_template('index.html', result=result, query=query)

@app.route("/index-documents")
def index_documents():
    return render_template('index-documents.html')

@app.post("/upload")
def handle_upload():
    # json = request.get_json()
    # filename = json['filename']
    file_uploaded = request.files['file']
    temp_path = '/tmp/'
    temp_file = os.path.join(temp_path, file_uploaded.filename)
    file_uploaded.save(temp_file)
    # rag_elastic.upload_file(temp_file)

    rag_elastic.addDocument(temp_file)
    os.remove(temp_file)
    return render_template('index-documents.html', filename='Uploaded')

# API ROUTES

@app.route("/chat", methods=['POST'])
def chat():
    json = request.get_json()
    message = json['message']
    session_id = json['session_id']
    if session_id == '':
        session_id = str(uuid4())
    result = rag_elastic.search_question(message, session_id)
    return jsonify({"message": result, "session_id": session_id}), 200

@app.route("/upload_docs", methods=['POST'])
def upload_docs():
    file_uploaded = request.files['file']
    temp_path = '/tmp/'
    temp_file = os.path.join(temp_path, file_uploaded.filename)
    file_uploaded.save(temp_file)
    # rag_elastic.upload_file(temp_file)

    rag_elastic.addDocument(temp_file)
    os.remove(temp_file)
    return jsonify({"status": "success"}), 200

@app.route("/re_index", methods=['POST'])
def re_index():
    json = request.get_json()
    session_id = json['session_id']
    rag_elastic.re_index(session_id)
    return jsonify({"status": "success"}), 200

