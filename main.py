from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store, get_embedding_model

def main():
    # Load the documents
    documents = load_documents("docs", with_preview=True)

    # Chunk the documents
    chunks = split_documents(documents, with_preview=True)
    
    # Embedding model 
    embedding_model = get_embedding_model(model_name="BAAI/bge-small-en-v1.5", device="cuda")

    # Create store
    vector_store = create_vector_store(embedding_model=embedding_model, persist_directory="db/chroma_db", overwrite=True)
    
    vector_store.add_documents(chunks)


if __name__ == "__main__":
    main()