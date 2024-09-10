from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

import os
import config

def create_vector_store():
    # Define the directory containing the HTML files
    directory_path = config.DOCUMENT_DIRECTORY_PATH

    # List to store all HTML data
    all_html_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # Load the HTML file using BSHTMLLoader
                loader = BSHTMLLoader(file_path)
                documents = loader.load()

                # Extract metadata and store HTML data
                metadata = {'filename': filename, 'source': file_path}
                for document in documents:
                    html_data = {
                        'filename': filename,
                        'metadata': metadata,
                        'page_content': document.page_content,
                        'source': file_path
                    }
                    all_html_data.append(html_data)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Check if any HTML data was loaded
    if not all_html_data:
        raise ValueError("No HTML data found. Check your HTML files and extraction logic.")

    # Convert HTML data into LangChain Document objects
    documents = [
        Document(
            page_content=data['page_content'],
            metadata=data['metadata']
        )
        for data in all_html_data
    ]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks found after splitting. Check your text splitting logic.")

    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model=config.embedding_deployment_name,
        api_key=config.OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT
    )

    # Load existing vector store or create new one
    vectordb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=azure_embedding)



    # Retrieve metadata of all documents in the vector store
    existing_metadata = set()
    for doc in vectordb._collection.get()['metadatas']:
        source = doc.get('source', '')
        if source:
            existing_metadata.add(source)

    # Filter out documents that are already in the vector store
    new_documents = [doc for doc in documents if doc.metadata.get('source', '') not in existing_metadata]

    # If no new documents, return existing vector store
    if not new_documents:
        print("No new documents to process.")
        return vectordb

    # Add new documents to the vector store
    vectordb.add_documents(new_documents)
    print(f"Processed {len(new_documents)} new documents.")
    
    return vectordb


