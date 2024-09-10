# # main.py

# import training
# import inference

# def main():
#     # Step 1: Create and save the vector store (Training Phase)
#     print("Starting training...")
#     db = training.create_vector_store()  # Create the vector store
#     db.save_local("faiss_index")  # Save FAISS index locally
#     print("Training completed and vector store saved.")

#     # Step 2: Perform inference (Inference Phase)
#     print("Starting inference...")
#     query = "What is Kibana?"
#     result = inference.perform_inference(query)
#     print("Inference result:", result)

# if __name__ == "__main__":
#     main()



# main.py
import training
import inference

def main():
    # Step 1: Create and save the vector store (Training Phase)
    print("Starting training...")
    db = training.create_vector_store()  # Create the vector store
    db.save_local("faiss_index")  # Save FAISS index locally
    print("Training completed and vector store saved.")

    # Step 2: Perform inference (Inference Phase)
    print("Starting inference...")
    query = "What is unstructured data?"
    result = inference.perform_inference(query)
    print("Inference result:", result)

if __name__ == "__main__":
    main()
