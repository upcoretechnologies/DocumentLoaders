import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import config

def extract_metadata_and_text(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            metadata = pdf.metadata
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""

        return {
            'title': metadata.get('Title', ''),
            'producer': metadata.get('Producer', ''),
            'author': metadata.get('Author', ''),
            'creation_date': metadata.get('CreationDate', ''),
            'page_content': text
        }
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        return {}

def create_vector_store():
    # List to store all PDFs' data
    all_pdfs_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(config.DOCUMENT_DIRECTORY_PATH):
        if filename.endswith('.pdf'):
            file_path = os.path.join(config.DOCUMENT_DIRECTORY_PATH, filename)
            
            try:
                # Extract metadata and text
                pdf_data = extract_metadata_and_text(file_path)
                pdf_data['filename'] = filename
                pdf_data['source'] = file_path
                
                all_pdfs_data.append(pdf_data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert to Document objects
    documents = [
        Document(
            page_content=data['page_content'],
            metadata={
                'title': data['title'],
                'producer': data['producer'],
                'author': data['author'],
                'creation_date': data['creation_date'],
                'source': data['source']
            }
        )
        for data in all_pdfs_data
    ]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model="upcoretext-embedding",   
        api_key=config.API_KEY,                   
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT    
    )

    # Create a Chroma vector store from documents
    vectordb = Chroma.from_documents(chunks, azure_embedding, persist_directory=config.PERSIST_DIRECTORY)


    existing_metadata = set()
    for doc in vectordb._collection.get()['metadatas']:
        source = doc.get('source', '')
        if source:
            existing_metadata.add(source)

    # print("Existing documents in the vector store:")
    # for source in existing_metadata:
    #     print(f" - {source}")

    # Filter out documents that are already in the vector store
    new_documents = [doc for doc in documents if doc.metadata.get('source', '') not in existing_metadata]

    # Debugging: Print new documents to be processed
    print(f"New documents to process: {len(new_documents)}")
    for doc in new_documents:
        print(f" - {doc.metadata.get('source', 'Unknown Source')}")

    if not new_documents:
        print("No new documents to process.")
        return vectordb


    existing_metadata = set()
    for doc in vectordb._collection.get()['metadatas']:
        source = doc.get('source', '')
        if source:
            existing_metadata.add(source)

    # # print("Existing documents in the vector store:")
    # for source in existing_metadata:
    #     print(f" - {source}")



    # Filter out documents that are already in the vector store
    new_documents = [doc for doc in documents if doc.metadata.get('source', '') not in existing_metadata]

    # Debugging: Print new documents to be processed
    print(f"New documents to process: {len(new_documents)}")
    for doc in new_documents:
        print(f" - {doc.metadata.get('source', 'Unknown Source')}")

    if not new_documents:
        print("No new documents to process.")
        return vectordb

    return vectordb
