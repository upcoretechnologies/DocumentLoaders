from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
OPENAI_API_KEY = "*****"
OPENAI_API_TYPE = "*****""
embedding_deployement_name = "*****"
OPENAI_API_VERSION = "*****"
AZURE_OPENAI_ENDPOINT = "*****"
llm_deployement_name = "*****"

# Step 1: Load and Chunk Documents
def read_doc(directory):
    loader = DirectoryLoader(directory, glob="*.*")
    documents = loader.load()
    return documents

docs = read_doc(r'C:/vector_databases/pin_cone_database/all_documents')
print(f"Total documents: {len(docs)}")

# Chunk the documents
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

chunks = chunk_data(docs)
print(f"Total chunks: {len(chunks)}")

# Step 2: Initialize Azure OpenAI Embeddings
azure_embedding = AzureOpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model=embedding_deployement_name,   
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Step 3: Initialize Pinecone and create index if it doesn't exist
pc = Pinecone(api_key="5f7689cd-75d0-46dc-9be6-b3cfe4d1b8a8")

index_name = "langchain-vectors"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1' 
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

#Creatingg PineconeVectorStore and Add Chunks
vector_store = PineconeVectorStore(index=index, embedding=azure_embedding)
vector_store.add_documents(documents=chunks)
print("Documents have been added to Pinecone vector store.")

# Initialize Azure OpenAI chat model
llm = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_deployment=llm_deployement_name,
    model="gpt-4o",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Set up retrieval QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Process a query
query = "What is Kibana"
result = qa_chain.invoke(query)

# Extract and print the plain text result
answer = result['result']
print(answer)
