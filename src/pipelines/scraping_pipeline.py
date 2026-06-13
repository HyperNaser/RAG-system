import os
import serpapi
import urllib3
import json

from .base import BasePipeline
from pathlib import Path
from pathvalidate import sanitize_filename

from typing import Any

class ScrapingPipeline(BasePipeline):
    def __init__(self, docs_path: str, engine: str = "google") -> None:
        self.docs_path = docs_path
        self.engine = engine
    
    def _execute(self):
        client = serpapi.Client(api_key=os.getenv("SERPAPI_KEY"), timeout=10)
        
        path = Path(self.docs_path)
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
                    "engine": self.engine,
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
        