from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import config
import os

def create_vector_store():
    # Load documents from the directory
    loader = DirectoryLoader(config.DOCUMENT_DIRECTORY_PATH, glob="*.*")
    documents = loader.load()

    # Initialize Azure OpenAI embeddings    
    azure_embedding = AzureOpenAIEmbeddings(
        model=config.embedding_deployment_name,   
        api_key=config.OPENAI_API_KEY,                   
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT    
    )

    # Check if the persistent directory exists, create if it doesn't
    if not os.path.exists(config.PERSIST_DIRECTORY):
        os.makedirs(config.PERSIST_DIRECTORY)

    # Load existing vector store if it exists
    vectordb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=azure_embedding)

    # Retrieve metadata of all documents in the vector store
    existing_metadata = set()
    if vectordb._collection.get():
        for doc in vectordb._collection.get()['metadatas']:
            source = doc.get('source', '')
            if source:
                existing_metadata.add(source)

    # Filter out documents that are already in the vector store
    new_documents = [doc for doc in documents if doc.metadata.get('source', '') not in existing_metadata]

    # Debugging: Print new documents to be processed
    print(f"New documents to process: {len(new_documents)}")
    for doc in new_documents:
        print(f" - {doc.metadata.get('source', 'Unknown Source')}")

    if not new_documents:
        print("No new documents to process.")
        return vectordb

    # Split new documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(new_documents)

    # Add chunk metadata for tracking
    for chunk in chunks:
        chunk.metadata['source'] = chunk.metadata.get('source', '')

    # Add new documents to the vector store with metadata
    vectordb.add_documents(chunks)

    print(f"Processed {len(new_documents)} new documents.")
    return vectordb