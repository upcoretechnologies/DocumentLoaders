# # inference.py
# from langchain_community.vectorstores import FAISS
# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# from langchain.chains import RetrievalQA
# import os

# # Define your settings here
# OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
# AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
# EMBEDDING_MODEL = "upcoretext-embedding"
# LLM_MODEL = "chatmodel"
# OPENAI_API_VERSION = "2024-05-01-preview"

# def perform_inference(query):
#     # Initialize Azure OpenAI embeddings
#     azure_embedding = AzureOpenAIEmbeddings(
#         model=EMBEDDING_MODEL,  # Use model for embeddings
#         api_key=OPENAI_API_KEY,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT
#     )

#     # Load the FAISS vector store
#     db = FAISS.load_local("faiss_index", azure_embedding, allow_dangerous_deserialization=True)
    
#     # Initialize Azure OpenAI chat model
#     llm = AzureChatOpenAI(
#         api_key=OPENAI_API_KEY,
#         model=LLM_MODEL,  # Specify the chat model
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_version=OPENAI_API_VERSION
#     )

#     # Set up retrieval QA chain
#     retriever = db.as_retriever(search_kwargs={"k": 2})
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

#     # Process a query
#     result = qa_chain.invoke(query)
#     return result

# if __name__ == "__main__":
#     query = "What is the impact of climate change on agriculture?"
#     result = perform_inference(query)
#     print("Inference result:", result)





# inference.py
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA

# Define your settings here
OPENAI_API_KEY = "bd27e26c317a45e580662ce658d890a9"
AZURE_OPENAI_ENDPOINT = "https://poc-slack.openai.azure.com/"
EMBEDDING_MODEL = "upcoretext-embedding"
LLM_MODEL = "chatmodel"
OPENAI_API_VERSION = "2024-05-01-preview"

def perform_inference(query):
    # Initialize Azure OpenAI embeddings
    azure_embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    # Load the FAISS vector store
    db = FAISS.load_local("faiss_index", azure_embedding, allow_dangerous_deserialization=True)
    
    # Initialize Azure OpenAI chat model
    llm = AzureChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION
    )

    # Set up retrieval QA chain
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Process a query
    result = qa_chain.invoke(query)

    # Format the result in a simple text format without query/answer structure
    answer = result['result']
    
    # Extract content from source documents
    sources = result['source_documents']
    formatted_sources = []
    
    for doc in sources:
        source_info = f"Source: {doc.metadata['source']}\n{doc.page_content}"
        formatted_sources.append(source_info)

    # Combine the final output
    final_output = f"{answer}\n\n" 
    
    return final_output

# if __name__ == "__main__":
#     query = "What is Kibana?"
#     result = perform_inference(query)
#     print(result)
