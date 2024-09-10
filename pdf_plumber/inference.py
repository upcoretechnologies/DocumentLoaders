from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
import config

def perform_inference(vectordb, query):
    # Initialize Azure ChatGPT model
    llm = AzureChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        azure_deployment=config.llm_deployment_name,  # Ensure this is a chat model deployment name
        model="gpt-4",  # Use a chat model compatible with the chatCompletion operation
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.OPENAI_API_VERSION
    )

    # Initialize retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Create the QA chain using RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    # Function to process and print LLM responses along with unique sources
    def process_llm_response(llm_response):
        print("Result:", llm_response['result'])
        
        # Use a set to track unique sources
        unique_sources = set()
        for source in llm_response["source_documents"]:
            if source.metadata['source'] not in unique_sources:
                unique_sources.add(source.metadata['source'])
                print("Source:", source.metadata['source'])

    # Execute query
    llm_response = qa_chain.invoke(query)
    process_llm_response(llm_response)



