import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import config
import os 

def create_vector_store():
    directory_path = config.DOCUMENT_DIRECTORY_PATH
    api_key = config.OPENAI_API_KEY
    embedding_deployement_name = config.embedding_deployment_name
    azure_endpoint = config.AZURE_OPENAI_ENDPOINT

    # List to store all PDFs' data
    all_pdfs_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            
            # Open and process the PDF file
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata

                # Extract content from each page
                pdf_content = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_content += page_text

                # Flatten and simplify metadata
                flat_metadata = {
                    'filename': filename,
                    'source': file_path,
                    'title': metadata.get('Title', ''),
                    'producer': metadata.get('Producer', ''),
                    'author': metadata.get('Author', ''),
                    'creation_date': metadata.get('CreationDate', ''),
                    'mod_date': metadata.get('ModDate', '')
                }

                # Collect the PDF data
                pdf_data = {
                    'filename': filename,
                    'metadata': flat_metadata,
                    'page_content': pdf_content,
                    'source': file_path
                }
                all_pdfs_data.append(pdf_data)



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

    

    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model=embedding_deployement_name,
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )


    # Define the directory to persist the vector store
    persist_directory = config.PERSIST_DIRECTORY

    # Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Create a Chroma vector store from documents
    try:
        db = Chroma.from_documents(chunks, azure_embedding, persist_directory=persist_directory)
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        raise

    return db


