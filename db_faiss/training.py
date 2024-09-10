# # training.py
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import AzureOpenAIEmbeddings
# import os

# # Define your settings here
# DOCUMENT_DIRECTORY_PATH = r"C:\vector_databases\all_documents"
# OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
# AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
# EMBEDDING_MODEL = "upcoretext-embedding"  # Use model instead of deployment_name

# def create_vector_store():
#     # Load documents
#     loader = DirectoryLoader(DOCUMENT_DIRECTORY_PATH, glob="*.*")
#     documents = loader.load()

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(documents)

#     # Initialize Azure OpenAI embeddings
#     azure_embedding = AzureOpenAIEmbeddings(
#         model=EMBEDDING_MODEL,  # Specify the embedding model name here
#         api_key=OPENAI_API_KEY,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT
#     )

#     # Create FAISS vector store
#     db = FAISS.from_documents(chunks, azure_embedding)
#     return db

# if __name__ == "__main__":
#     db = create_vector_store()
#     db.save_local("faiss_index")  # Save FAISS index locally
#     print("Vector store created and saved.")


# training.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Define your settings here
DOCUMENT_DIRECTORY_PATH = r"C:\vector_databases\all_documents"
OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
EMBEDDING_MODEL = "upcoretext-embedding"

def create_vector_store():
    # Load documents
    loader = DirectoryLoader(DOCUMENT_DIRECTORY_PATH, glob="*.*")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    # Create FAISS vector store
    db = FAISS.from_documents(chunks, azure_embedding)
    return db

# if __name__ == "__main__":
#     db = create_vector_store()
#     db.save_local("faiss_index")  # Save FAISS index locally
#     print("Vector store created and saved.")
