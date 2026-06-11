from src.vector_store import get_embedding_model
from src.pipelines import run_ingestion_pipeline

def main():
    embedding_model = get_embedding_model(model_name="BAAI/bge-small-en-v1.5", device="cuda")
    run_ingestion_pipeline(docs_path="docs", embedding_model=embedding_model, with_previews=True, overwrite=True)

if __name__ == "__main__":
    main()