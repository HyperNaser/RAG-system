from langchain_core.vectorstores.base import VectorStore

def query_vector_store(query_text: str, vector_store: VectorStore, k: int = 3):
    """Connects to the database and retrieves the top K most relevant document chunks"""
    print(f"Searching database for: '{query_text}'...")
    
    results = vector_store.similarity_search(query_text, k=k)
    
    if not results:
        print("No matching context found in the database.")
        return
        
    print(f"Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results):
        print(f"\nMatch {i+1} (Source: {doc.metadata.get('filename')}):")
        print(f"   {doc.page_content}...")
        
    return results