from training import create_vector_store
from inference import perform_inference
import config

def main():
    # Create vector store (only needed once, unless documents are updated)
    vectordb = create_vector_store()

    # Run inference (you can call this multiple times with different queries)
    query = "What is elasticsearch?"
    perform_inference(vectordb, query)

if __name__ == "__main__":
    main()
