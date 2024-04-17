from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# loader = PyPDFLoader('./Blockchain.pdf')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()


from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings()

elastic_vector_search = ElasticsearchStore(
    es_url="http://192.168.100.5:9200",
    index_name="test-dos",
    embedding=embedding
)

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

def make_search(query):
    elastic_vector_search.similarity_search(query)
    retriever = elastic_vector_search.as_retriever(search_kwargs={"k": 4})
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | ChatOpenAI() 
        | StrOutputParser()
    )
    print(query)
    return chain.invoke(query)

def upload_file(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    ElasticsearchStore.from_documents(
        docs,
        embeddings,
        es_url="http://192.168.100.5:9200",
        index_name="test-dos",
    )
