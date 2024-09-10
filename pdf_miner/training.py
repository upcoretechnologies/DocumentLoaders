import os
from pdfminer.high_level import extract_text
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import config

def create_vector_store():
    # Directory path from config
    directory_path = config.DIRECTORY_PATH
    api_key = config.OPENAI_API_KEY
    embedding_deployment_name = config.embedding_deployment_name
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT

    # List to store all PDFs' data
    all_pdfs_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # Extract text from the PDF file using pdfminer
                pdf_content = extract_text(file_path)

                # Flatten and simplify metadata
                flat_metadata = {
                    'filename': filename,
                    'source': file_path,
                    'title': '',  # pdfminer does not extract metadata
                    'producer': '',  # pdfminer does not extract metadata
                    'author': '',  # pdfminer does not extract metadata
                    'creation_date': '',  # pdfminer does not extract metadata
                    'mod_date': ''  # pdfminer does not extract metadata
                }

                # Collect the PDF data
                pdf_data = {
                    'filename': filename,
                    'metadata': flat_metadata,
                    'page_content': pdf_content,
                    'source': file_path
                }
                all_pdfs_data.append(pdf_data)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert to Document objects
    documents = [
        Document(
            page_content=data['page_content'],
            metadata=data['metadata']
        )
        for data in all_pdfs_data
    ]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Extract texts and metadatas from chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model=config.embedding_deployment_name,
        api_key=config.OPENAI_API_KEY,
        azure_endpoint=azure_endpoint
    )

    # Define the directory to persist the vector store
    persist_directory = config.PERSIST_DIRECTORY

    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Create a Chroma vector store from documents without explicit IDs
    try:
        db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=azure_embedding, persist_directory=persist_directory)
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        raise

    return db
