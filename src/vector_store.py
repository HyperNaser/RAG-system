from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str, device: str = "cuda") -> Embeddings:
    """Initializes local embedding model"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )

def create_vector_store(embedding_model: Embeddings, persist_directory: str, overwrite: bool = False):
    """Create and persist ChromaDB vector store"""
    
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if overwrite:
        existing_data = vector_store.get()
        record_count = len(existing_data.get("ids", []))

        if record_count > 0:
            print("Clearing records from vector storage")
            vector_store.reset_collection()
        else:
            print("Vector storage is already empty")
    
    return vector_store