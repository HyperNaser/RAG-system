import argparse
from dotenv import load_dotenv
from src.config import config


def main():
    parser = argparse.ArgumentParser(description="RAG Application Interface")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", action="store_true", help="Run the ingestion pipeline to populate the DB")
    group.add_argument("--query", action="store_true", help="Launch interactive search mode")
    group.add_argument("--scrape", action="store_true", help="Launch interactive google search scraper")

    args = parser.parse_args()

    if args.scrape:
        from src.pipelines.scraping_pipeline import ScrapingPipeline
        ScrapingPipeline(docs_path=config.DOCS_PATH).run()
        return

    print("Initializing embedding model and vector store context...")
    from src.vector_store import get_embedding_model, create_chromadb_vector_store
    embedding_model = get_embedding_model(model_name=config.MODEL_NAME, device=config.DEVICE)
    vector_store = create_chromadb_vector_store(embedding_model=embedding_model, persist_directory=config.DB_PATH)

    if args.ingest:
        from src.pipelines.ingestion_pipeline import IngestionPipeline
        IngestionPipeline(vector_store=vector_store, docs_path=config.DOCS_PATH, with_previews=True, overwrite=True).run()
    elif args.query:
        from src.pipelines.retrieval_pipeline import RetrievalPipeline
        RetrievalPipeline(vector_store=vector_store, app_config=config).run()

if __name__ == "__main__":
    load_dotenv()
    main()