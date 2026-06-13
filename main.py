import argparse
from dotenv import load_dotenv
from src.vector_store import get_embedding_model, create_chromadb_vector_store
from src.config import config
from src import pipelines


def main():
    parser = argparse.ArgumentParser(description="RAG Application Interface")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", action="store_true", help="Run the ingestion pipeline to populate the DB")
    group.add_argument("--query", action="store_true", help="Launch interactive search mode")
    group.add_argument("--scrape", action="store_true", help="Launch interactive google search scraper")

    args = parser.parse_args()

    embedding_model = get_embedding_model(model_name=config.MODEL_NAME, device=config.DEVICE)
    vector_store = create_chromadb_vector_store(embedding_model=embedding_model, persist_directory=config.DB_PATH)

    if args.ingest:
        pipelines.run_ingestion_pipeline(vector_store=vector_store, docs_path=config.DOCS_PATH, with_previews=True, overwrite=True)
    elif args.query:
        pipelines.run_retrieval_pipeline(vector_store=vector_store)
    elif args.scrape:
        pipelines.run_scrape_pipeline(docs_path=config.DOCS_PATH)

if __name__ == "__main__":
    load_dotenv()
    main()