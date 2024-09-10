# main.py

from training import create_vector_store
from inference import perform_inference

def main():
    # Create vector store (only needed once, unless documents are updated)
    vectordb = create_vector_store()

   
   
    query = "How to install apm?"
    perform_inference(vectordb, query)

if __name__ == "__main__":
    main()
