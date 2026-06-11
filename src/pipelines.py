from langchain_core.vectorstores.base import VectorStore
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.retriever import query_vector_store

def run_ingestion_pipeline(vector_store: VectorStore, docs_path: str,  with_previews: bool = False, overwrite: bool = False):
    """Starts the ingestion pipeline"""
    documents = load_documents(docs_path, with_preview=with_previews)

    chunks = split_documents(documents, with_preview=with_previews)
    
    if overwrite:
        print("Resetting Vector Store...")
        if hasattr(vector_store, "reset_collection"):
            getattr(vector_store, "reset_collection")()
        else:
            success = vector_store.delete()
            if not success:
                raise NotImplementedError("Provided Vector Store does not implement the delete() function correctly and some elements might be left")
        
        print("Successfully deleted all records")

    print(f"Adding {len(chunks)} chunks to Vector Store...")
    vector_store.add_documents(chunks)
    print("Successfully added all chunks.")

def run_retrieval_pipeline(vector_store: VectorStore):
    """Starts an interactive RAG query loop"""
    
    print("\nInteractive RAG Session Started! (Type 'exit', 'q' or 'quit' to stop)")
    print("=" * 60)
    
    while True:
        user_query = input("\nEnter your search query: ").strip()
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nClosing interactive retrieval session. Goodbye!")
            break
            
        if not user_query:
            continue
            
        query_vector_store(
            query_text=user_query,
            vector_store=vector_store,
            k=3,
        )

        print("=" * 60)
