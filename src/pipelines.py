import os
import serpapi
import urllib3
import json

from pathlib import Path
from pathvalidate import sanitize_filename
from langchain_core.vectorstores.base import VectorStore
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.retriever import query_vector_store

from typing import Any

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

def run_scrape_pipeline(docs_path: str):
    client = serpapi.Client(api_key=os.getenv("SERPAPI_KEY"), timeout=10)
    
    path = Path(docs_path)
    http = urllib3.PoolManager()
    
    print("\nInteractive Google Scraper Session Started! (Type 'exit', 'q' or 'quit' to stop)")
    print("=" * 60)
    try:
        while True:
            query = input("Search: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\nClosing google scraping session. Goodbye!")
                break

            results = client.search({ #type: ignore
                "engine": "google",
                "q": query,
            })

            if isinstance(results, serpapi.SerpResults):
                print(f"Got {len(results)} SerpResults!")
                organic_results = results.get("organic_results") #type: ignore
                if isinstance(organic_results, list):
                    for result in organic_results: #type: ignore
                        if isinstance(result, dict):
                            title: str = result.get("title") #type: ignore
                            link: str = result.get("link") #type: ignore
                            metadata: dict[str, Any] = result.get("about_this_result") #type: ignore
                            
                            file_path = path / (sanitize_filename(title) + ".json")

                            try:
                                response = http.request("GET", link, timeout=10.0)
                                html_text = response.data.decode("utf-8")
                                
                                page_info = { #type: ignore
                                    "link": link,
                                    "title": title,
                                    "metadata": metadata,
                                    "raw_html": html_text
                                }
                                
                                with open(file_path, "w", encoding="utf-8") as file:
                                    json.dump(page_info, file, indent=2, ensure_ascii=False)
                                
                                print(f"Saved successfully to: {file_path}")

                            except Exception as e:
                                print(f"Error fetching the page for {title} from {link}: {e}")
                            
            elif isinstance(results, str):
                print(results)
            else:
                print("UNKNOWN TYPE")

    except serpapi.HTTPError as e:
        if e.status_code == 401: # Invalid API key
            print(e.error) # "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"
        elif e.status_code == 400: # Missing required parameter
            print(e.error)
        elif e.status_code == 429: # Exceeds the hourly throughput limit OR account run out of searches
            print(e.error)
    except serpapi.TimeoutError as e:
        print(f"The request timed out: {e}")