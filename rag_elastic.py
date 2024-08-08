import os
from langchain_elasticsearch import ElasticsearchStore, ElasticsearchChatMessageHistory
from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

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
    ELASTIC_HISTORY_INDEX = 'rag-history'
    #template_prompt = """Responde la pregunta basándote solo en el siguiente contexto:

    def __init__(self):
        self.es_client = Elasticsearch(
            cloud_id=os.environ['ELASTIC_CLOUD_ID'],
            api_key=os.environ['ELASTIC_API_KEY'],
        )
        # Local docker connection
        # self.es_client = Elasticsearch('http://localhost:9200')
        embedding = OpenAIEmbeddings()
        self.elastic_vector_search = ElasticsearchStore(
            index_name=self.ELASTIC_INDEX,
            es_connection=self.es_client,
            embedding=embedding,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )

        # self.template_prompt = """Answer the question based only on the following context:

        self.template_prompt = """Answer the question based on the following context:

        Context:
        {context}

        Question: {question}
        """

        self.propmt_history = """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat history:
        {context}

        Follow Up Question: {{ question }}
        Standalone question:
        """

    def get_chat_history(self, session_id):
        return ElasticsearchChatMessageHistory(
            es_connection=self.es_client, index=self.ELASTIC_HISTORY_INDEX, session_id=session_id
    )

    def get_chat_context(self, question, chat_history):
        messages = ""
        for message in chat_history.messages:
            if message.type == 'human':
                messages += f"Question: {message.content}\n"
            elif message.type == 'ai':
                messages += f"Response: {message.content}\n"
        return messages
    
    def get_condensed_question(self, question, chat_history):
        context = self.get_chat_context(question, chat_history)
        prompt = self.propmt_history.replace("{context}", context)
        prompt = prompt.replace("{question}", question)
        chain = ChatOpenAI()

        print(prompt)

        return chain.invoke(prompt).content

    def make_search(self, query):
        # self.elastic_vector_search.similarity_search(query)
        retriever = self.elastic_vector_search.as_retriever(search_kwargs={"k": 4})
        prompt = ChatPromptTemplate.from_template(self.template_prompt)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | ChatOpenAI() 
            | StrOutputParser()
        )
        
        return chain.invoke(query)
    
    def search_question(self, question, session_id):
        chat_history = self.get_chat_history(session_id)

        if len(chat_history.messages):
            condensed_question = self.get_condensed_question(question, chat_history)
            print(condensed_question)
        else:
            condensed_question = question
        
        retriever = self.elastic_vector_search.as_retriever(search_kwargs={"k": 4})
        print(retriever.invoke(condensed_question))
        # retriever = self.elastic_vector_search.similarity_search(condensed_question, k=10)
        prompt = ChatPromptTemplate.from_template(self.template_prompt)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | ChatOpenAI() 
        )

        response = chain.invoke(condensed_question)
        chat_history.add_user_message(question)
        chat_history.add_ai_message(response)
        return response.content

    def save_documents(self, documents):
        # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        # docs = text_splitter.split_documents(documents)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=20,
            length_function=len,
            keep_separator=True,
            separators=[
                "\n \n",
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                " ",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )

        docs = text_splitter.split_documents(documents)
        self.elastic_vector_search.add_documents(docs)

    def addDocument(self, filename):
        if filename.lower().endswith('pdf'):
            docs = PyPDFLoader(filename).load()
        if filename.lower().endswith('json'):
            docs = JSONLoader(
                file_path = filename,
                jq_schema = os.getenv("json_schema"),
                text_content = os.getenv("json_text_content") == "True",
            ).load()
        if filename.lower().endswith('csv'):
            docs = CSVLoader(filename).load()
        if filename.lower().endswith('docx'):
            docs = Docx2txtLoader(filename).load()
        if filename.lower().endswith('xlsx'):
            docs = UnstructuredExcelLoader(filename).load()
        if filename.lower().endswith('pptx'):
            docs = UnstructuredPowerPointLoader(filename).load()

        return self.save_documents(docs)
    
    def re_index(self, session_id = None):
        self.es_client.indices.delete(index=self.ELASTIC_INDEX)

        if session_id:
            self.get_chat_history(session_id).clear()
        
# perplexity
# add pipes from langchain to internet if not have context
