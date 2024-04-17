from flask import Flask, render_template, request
import os

app = Flask(__name__)
from rag_elastic import RagElastic

rag_elastic = RagElastic()

@app.route("/")
def index():
    return render_template('index.html')

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
    file_uploaded = request.files['file']
    temp_path = '/tmp/'
    temp_file = os.path.join(temp_path, file_uploaded.filename)
    file_uploaded.save(temp_file)
    rag_elastic.upload_file(temp_file)
    os.remove(temp_file)
    return render_template('index-documents.html', filename=temp_file)