from langchain_core.embeddings import Embeddings
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store

def run_ingestion_pipeline(docs_path: str, embedding_model: Embeddings, with_previews: bool = False, overwrite: bool = False):
    documents = load_documents(docs_path, with_preview=with_previews)

    chunks = split_documents(documents, with_preview=with_previews)
    
    vector_store = create_vector_store(embedding_model=embedding_model, persist_directory="db/chroma_db", overwrite=overwrite)
    
    vector_store.add_documents(chunks)
