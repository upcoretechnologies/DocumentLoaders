from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
import config

def perform_inference(vectordb, query):
    # Initialize Azure ChatGPT model
    llm = AzureChatOpenAI(
      api_key=config.API_KEY,
        azure_deployment=config.llm_deployment_name,
        model="gpt-4o",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.OPENAI_API_VERSION
    )

    # Initialize retriever from the vector store
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 12})

    # Create the QA chain using RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Function to process and print LLM responses along with unique sources
    def process_llm_response(llm_response):
        print("Answer:", llm_response['result'])
        
        # Use a set to track unique sources
        unique_sources = set()
        for source in llm_response["source_documents"]:
            if source.metadata['source'] not in unique_sources:
                unique_sources.add(source.metadata['source'])
                print("Source:", source.metadata['source'])

    # Run the query
    llm_response = qa_chain.invoke(query)
    process_llm_response(llm_response)

