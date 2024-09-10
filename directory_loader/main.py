from training import create_vector_store
from inference import perform_inference
import config

def main():
    # Create vector store (only needed once, unless documents are updated)
    vectordb = create_vector_store()

    # Run inference (you can call this multiple times with different queries)
    # query = "What is unstrucrured data?"
    # query="How to process the unstructured data?"
    # query="What is elasticsearch?"
    # query="What are uses of elasticsearch?"
    # query="What are the use cases of databricks?"
    # query="How to install databricks?"
    # query="How to track the application process?"
    # query="What are the challenegs with unstructured data?"
    # query="What do the case studies  tell about the  unstructured data"
    # query="What is canvas?"
    # query="one which localhost does kibana runs?"
    # query="How to perform time series dataquery in kibana?"
    # query="How do kibana do geography location analysis?"
    # query ="Give the features of kibana"
    # query="What information do you get from sample pdf"
    # query="How to get credentials of elasticsearch?"
    # query="Wow to do text analytics and machine learning on unstructured data?"
    # query="How to integrate structure data with unstructure data?"
    # query="What rae the benefits of analysis of unstructured data?"
    # query="How to do indfexing in elasticsearch?"
    # query ="How many countries are listed in the list , list the name of the countries?"
    # query="What is the name of the country that has highest economic equality?"
    query="On which indicators the countries are scored , list the name of the indicators?"
    # perform_inference(vectordb, query)
    # query = " what information is in the docvument?"
    perform_inference(vectordb, query)

if __name__ == "__main__":
    main()
