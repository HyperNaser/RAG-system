from .base import BasePipeline
from langchain_core.vectorstores.base import VectorStore
from src.document_loader import load_documents
from src.text_splitter import split_documents

class IngestionPipeline(BasePipeline):
    def __init__(self, vector_store: VectorStore, docs_path: str,  with_previews: bool = False, overwrite: bool = False) -> None:
        self.vector_store = vector_store
        self.docs_path = docs_path
        self.with_previews = with_previews
        self.overwrite = overwrite
        
    def _execute(self):
        """Starts the ingestion pipeline"""
        documents = load_documents(self.docs_path, with_preview=self.with_previews)

        chunks = split_documents(documents, with_preview=self.with_previews)
        
        if self.overwrite:
            print("Resetting Vector Store...")
            if hasattr(self.vector_store, "reset_collection"):
                getattr(self.vector_store, "reset_collection")()
            else:
                success = self.vector_store.delete()
                if not success:
                    raise NotImplementedError("Provided Vector Store does not implement the delete() function correctly and some elements might be left")
            
            print("Successfully deleted all records")

        print(f"Adding {len(chunks)} chunks to Vector Store...")
        self.vector_store.add_documents(chunks)
        print("Successfully added all chunks.")