from .base import BasePipeline
from langchain_core.vectorstores.base import VectorStore
from src.retriever import query_vector_store

class RetrievalPipeline(BasePipeline):
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
    
    def _execute(self):
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
                vector_store=self.vector_store,
                k=3,
            )

            print("=" * 60)