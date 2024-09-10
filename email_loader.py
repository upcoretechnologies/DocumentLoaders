import os
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Define the scopes for Gmail API
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Azure OpenAI configuration
OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
OPENAI_API_TYPE = "azure"
embedding_deployement_name = "upcoretext-embedding"
OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
llm_deployement_name = "chatmodel"

def load_credentials():
    """Load credentials from file and handle authorization."""
    creds = None
    
    # Check if token.json exists
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If no valid credentials, handle authorization flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
        if not creds:
            try:
                # Start the authorization flow
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",  
                    SCOPES,
                )
                creds = flow.run_local_server(port=0)
                
                # Save the new credentials to token.json
                with open("token.json", "w", encoding='utf-8') as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"Error during authorization: {e}")
                return None
    
    return creds

def fetch_emails(creds):
    """Fetch emails from Gmail using the provided credentials."""
    try:
        # Build the Gmail service
        service = build('gmail', 'v1', credentials=creds)
        
        # List messages from the user's inbox
        results = service.users().messages().list(userId='me', maxResults=50).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print('No messages found.')
        else:
            # print('Messages:')
            email_texts = []
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                snippet = msg.get('snippet', '')
                # print(f"Message ID: {msg['id']}")
                # print(f"Snippet: {snippet}")
                email_texts.append(snippet)
            
            # Process emails with the RAG model
            process_emails_with_rag(email_texts)
    
    except Exception as e:
        print(f"An error occurred: {e}")

def process_emails_with_rag(email_texts):
    """Process email texts with the RAG model."""



    # Create a temporary directory to store email documents
    temp_dir = "temp_email_documents"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save email snippets as temporary files
    for i, text in enumerate(email_texts):
        try:
            with open(os.path.join(temp_dir, f"email_{i}.txt"), "w", encoding='utf-8') as f:
                f.write(text)
        except UnicodeEncodeError as e:
            print(f"Encoding error while writing file {i}: {e}")
            continue
    
    # Load documents from the temporary directory
    try:
        loader = DirectoryLoader(temp_dir, glob="*.*")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    # Split documents into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return
    
    # Initialize Azure OpenAI embeddings
    try:
        azure_embedding = AzureOpenAIEmbeddings(
            model=embedding_deployement_name,   
            api_key=OPENAI_API_KEY,                   
            azure_endpoint=AZURE_OPENAI_ENDPOINT    
        )
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return

    # Initialize the LLM
    try:
        llm = AzureChatOpenAI(
            api_key=OPENAI_API_KEY,
            azure_deployment=llm_deployement_name,
            model="gpt-4o",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=OPENAI_API_VERSION
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # Create a FAISS vector store and retriever
    try:
        db = FAISS.from_documents(chunks, azure_embedding)
        retriever = db.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Error creating vector store or retriever: {e}")
        return

    # Crreating a RetrievalQA chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return
    


# passing the query 
    query = "What was the last  mail??"
    try:
        query_ans = qa_chain.invoke(query)
        print("RAG Model Response:", query_ans)
    except Exception as e:
        print(f"Error running query: {e}")
    
    # Clean up temporary directory
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temporary directory: {e}")

def main():
    creds = load_credentials()
    if creds:
        fetch_emails(creds)
    else:
        print("Failed to load credentials.")

if __name__ == "__main__":
    main()




























# import os
# import requests
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build

# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain_openai import AzureChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # Define the scopes for Gmail API
# SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# # Azure OpenAI configuration
# OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
# OPENAI_API_TYPE = "azure"
# embedding_deployement_name = "upcoretext-embedding"
# OPENAI_API_VERSION = "2024-05-01-preview"
# AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
# llm_deployement_name = "chatmodel"

# # Document directory path (for testing purposes)
# DOCUMENT_DIRECTORY_PATH = r'C:\Users\HP\Dropbox\PC\Downloads\my_project\dir_loader\all_documents'

# def load_credentials():
#     """Load credentials from file and handle authorization."""
#     creds = None
    
#     # Check if token.json exists
#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
#     # If no valid credentials, handle authorization flow
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             try:
#                 creds.refresh(Request())
#             except Exception as e:
#                 print(f"Error refreshing credentials: {e}")
#                 creds = None
#         if not creds:
#             try:
#                 # Start the authorization flow
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     "credentials.json",  # Ensure this file exists
#                     SCOPES,
#                 )
#                 creds = flow.run_local_server(port=0)
                
#                 # Save the new credentials to token.json
#                 with open("token.json", "w", encoding='utf-8') as token:
#                     token.write(creds.to_json())
#             except Exception as e:
#                 print(f"Error during authorization: {e}")
#                 return None
    
#     return creds

# def fetch_emails(creds):
#     """Fetch emails from Gmail using the provided credentials."""
#     try:
#         # Build the Gmail service
#         service = build('gmail', 'v1', credentials=creds)
        
#         # List messages from the user's inbox
#         results = service.users().messages().list(userId='me', maxResults=50).execute()
#         messages = results.get('messages', [])
        
#         if not messages:
#             print('No messages found.')
#         else:
#             print('Fetching emails...')
#             email_texts = []
#             for message in messages:
#                 msg = service.users().messages().get(userId='me', id=message['id']).execute()
#                 snippet = msg.get('snippet', '')
#                 email_texts.append(snippet)
            
#             # Process emails with the RAG model
#             if email_texts:
#                 print("Processing emails with RAG...")
#                 process_emails_with_rag(email_texts)
#             else:
#                 print("No email snippets to process.")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")

# def process_emails_with_rag(email_texts):
#     """Process email texts with the RAG model."""
#     temp_dir = "temp_email_documents"
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # Save email snippets as temporary files
#     for i, text in enumerate(email_texts):
#         try:
#             with open(os.path.join(temp_dir, f"email_{i}.txt"), "w", encoding='utf-8') as f:
#                 f.write(text)
#         except UnicodeEncodeError as e:
#             print(f"Encoding error while writing file {i}: {e}")
#             continue
    
#     # Load documents from the temporary directory
#     try:
#         loader = DirectoryLoader(temp_dir, glob="*.*")
#         documents = loader.load()
#         print(f"Loaded {len(documents)} documents.")
#         if not documents:
#             print("No documents loaded. Check the source directory.")
#     except Exception as e:
#         print(f"Error loading documents: {e}")
#         return
    
#     # Split documents into chunks
#     try:
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = text_splitter.split_documents(documents)
#         print(f"Number of chunks: {len(chunks)}")
#         if not chunks:
#             print("No chunks created. Check the splitting configuration.")
#         for chunk in chunks[:5]:  # Print first 5 chunks for inspection
#             print(chunk)
#     except Exception as e:
#         print(f"Error splitting documents: {e}")
#         return
    
#     # Initialize Azure OpenAI embeddings
#     try:
#         azure_embedding = AzureOpenAIEmbeddings(
#             model=embedding_deployement_name,   
#             api_key=OPENAI_API_KEY,                   
#             azure_endpoint=AZURE_OPENAI_ENDPOINT    
#         )
#     except Exception as e:
#         print(f"Error initializing embeddings: {e}")
#         return

#     # Initialize the LLM
#     try:
#         llm = AzureChatOpenAI(
#             api_key=OPENAI_API_KEY,
#             azure_deployment=llm_deployement_name,
#             model="gpt-4o",
#             azure_endpoint=AZURE_OPENAI_ENDPOINT,
#             api_version=OPENAI_API_VERSION
#         )
#     except Exception as e:
#         print(f"Error initializing LLM: {e}")
#         return

#     # Create a FAISS vector store and retriever
#     try:
#         db = FAISS.from_documents(chunks, azure_embedding)
#         print(f"FAISS vector store created with {len(db)} entries.")
#         if len(db) == 0:
#             print("FAISS vector store is empty. Check the embedding process.")
#         retriever = db.as_retriever(search_kwargs={"k": 2})
#     except Exception as e:
#         print(f"Error creating vector store or retriever: {e}")
#         return

#     # Create a RetrievalQA chain
#     try:
#         qa_chain = RetrievalQA.from_chain_type(
#             llm, 
#             chain_type="stuff", 
#             retriever=retriever, 
#             return_source_documents=True
#         )
#     except Exception as e:
#         print(f"Error creating QA chain: {e}")
#         return
    
#     # Example query
#     query = "What are the installation steps of Elasticsearch?"
#     try:
#         query_ans = qa_chain.run(query)
#         print("RAG Model Response:", query_ans)
#     except Exception as e:
#         print(f"Error running query: {e}")
    
#     # Clean up temporary directory
#     try:
#         for file in os.listdir(temp_dir):
#             os.remove(os.path.join(temp_dir, file))
#         os.rmdir(temp_dir)
#     except Exception as e:
#         print(f"Error cleaning up temporary directory: {e}")

# def main():
#     creds = load_credentials()
#     if creds:
#         fetch_emails(creds)
#     else:
#         print("Failed to load credentials.")

# if __name__ == "__main__":
#     main()
