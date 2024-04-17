import os
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# db = ElasticsearchStore.from_documents(
#     docs,
#     embeddings,
#     es_url="http://192.168.100.5:9200",
#     index_name="test-basic",
# )

# db.client.indices.refresh(index="test-basic")

# query = 'How blockchain could revolutionize the internet of things'
# results = elastic_vector_search.similarity_search(query)
# retriever = elastic_vector_search.as_retriever(search_kwargs={"k": 4})
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()} 
#     | prompt 
#     | ChatOpenAI() 
#     | StrOutputParser()
# )

# print(chain.invoke("que son los nodos mineros"))

# db.client.indices.refresh(index="test-basic")

# query = 'How blockchain could revolutionize the internet of things'
# results = db.similarity_search(query)
# print(results)

class RagElastic:
    ELASTIC_INDEX = 'rag-documents'
    template_prompt = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    def __init__(self):
        self.es_client = Elasticsearch(
            cloud_id=os.environ['ELASTIC_CLOUD_ID'],
            api_key=os.environ['ELASTIC_API_KEY'],
        )
        # Local docker connection
        # self.es_client = Elasticsearch('http://localhost:9200')
        embedding = OpenAIEmbeddings()
        self.elastic_vector_search = ElasticsearchStore(
            index_name="test_index",
            es_connection=self.es_client,
            embedding=embedding,
        )

    def make_search(self, query):
        self.elastic_vector_search.similarity_search(query)
        retriever = self.elastic_vector_search.as_retriever(search_kwargs={"k": 4})
        prompt = ChatPromptTemplate.from_template(self.template_prompt)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | ChatOpenAI() 
            | StrOutputParser()
        )
        
        return chain.invoke(query)

    def upload_file(self, file):
        loader = PyPDFLoader(file)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        self.elastic_vector_search.from_documents(
            docs,
            embeddings,
            index_name=self.ELASTIC_INDEX,
        )

