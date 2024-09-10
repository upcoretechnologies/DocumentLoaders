from training import create_vector_store
from inference import perform_inference

def main():
    vectordb = create_vector_store()

    # query = "What are the topics discussed in the documents?"
    # query="What are the freatures of unstructured data?"
    # query="What is inverted index"
    # query="How to do searching through elasticsearch?"
    # query="What are the use cases of elasticseach"
    # query="How to connect elasticsearch with langchain"
    query="What is the pricing of elasticsearch ?"
    perform_inference(vectordb, query)

if __name__ == "__main__":
    main()