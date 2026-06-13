# Local RAG Experimentation Platform

A modular, local Retrieval-Augmented Generation (RAG) backend built in Python. This repository ingests local documents, converts them to Markdown, splits text into retrievable chunks, generates local embeddings, and stores them in a persistent Chroma vector database. It then uses a local LLM (via Ollama) to generate contextually-aware answers based on retrieved documents.

## Features

* Local document ingestion using `docling`
* Markdown conversion and metadata extraction
* Chunking with `langchain_text_splitters` (`chunk_size=500`, `chunk_overlap=0`)
* Local embedding generation with `HuggingFaceEmbeddings`
* Persistent vector storage with `Chroma`
* Local LLM inference via Ollama for RAG-powered question answering
* Interactive CLI retrieval session with contextual answer generation
* Web scraping via Google Search (powered by SerpAPI) to populate vector store

## Prerequisites

* Python environment with dependencies installed from `requirements.txt`
* A `docs/` folder containing the source documents to ingest
* A local GPU if you want to use the default `device="cuda"` setting
* `db/chroma_db` is used to persist the vector store between runs
* **Ollama** running locally (default: `http://localhost:11434`) for LLM inference
* (Optional) `SERPAPI_KEY` environment variable for web scraping via Google Search

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Add your documents to the `docs/` directory, or use the scraper to fetch documents from the web.
2. (Optional) Use web scraping to gather documents:

```bash
python main.py --scrape
```

3. Build the vector store:

```bash
python main.py --ingest
```

4. Start the interactive retrieval session:

```bash
python main.py --query
```

5. In query mode, type a question and press Enter. Type `exit`, `quit`, or `q` to stop.

## Configuration

The project can be configured via `config.toml`:

```toml
[embedding]
model_name = "BAAI/bge-small-en-v1.5"
device = "cuda"

[llm]
model_name = "qwen2.5:3b"
base_url = "http://localhost:11434"
num_ctx = 4096

[paths]
db_path = "db/chroma_db"
docs_path = "docs"
```

If `config.toml` is missing, the application uses default values (as shown above). You can override any setting by editing the config file.

**LLM Configuration Notes:**
* `model_name`: The Ollama model to use for inference (e.g., `qwen2.5:3b`, `llama2`, `mistral`)
* `base_url`: The Ollama server URL (default: `http://localhost:11434`)
* `num_ctx`: Context window size for the LLM (default: `4096`)

## How it works

### `main.py`

* Loads configuration from `config.toml` (or uses defaults)
* Instantiates a Hugging Face embedding model based on config
* Creates a persistent Chroma vector store
* Initializes the local LLM via Ollama when needed
* Runs the ingestion, retrieval, or scraping pipeline based on CLI flags

### `src/pipelines/base.py`

* Defines `BasePipeline` abstract base class
* All pipelines inherit from this class and implement the `_execute()` method
* The `run()` method serves as the public interface for executing pipelines

### `src/pipelines/ingestion_pipeline.py`

* `IngestionPipeline` class that:
  * Loads documents from `docs/`
  * Splits them into chunks
  * Optionally resets the vector store when `overwrite=True`
  * Adds chunks to the Chroma database

### `src/pipelines/retrieval_pipeline.py`

* `RetrievalPipeline` class that:
  * Initializes the local LLM and builds a RAG chain
  * Starts an interactive loop
  * For each query: retrieves the top 3 most relevant chunks and generates contextual answers
  * Uses a prompt template to ensure the LLM answers only from retrieved context
  * Displays both the retrieved sources and generated answers

### `src/pipelines/scraping_pipeline.py`

* `ScrapingPipeline` class that:
  * Uses SerpAPI to search for web results (Google is the default engine)
  * Supports multiple search engines via the `engine` parameter (customizable)
  * Saves search results as JSON files to the `docs/` directory
  * Requires `SERPAPI_KEY` environment variable to be set
  * Provides an interactive search interface to populate documents from the web

### `src/config.py`

* Loads configuration from `config.toml`
* Provides an `AppConfig` dataclass with embedding settings, LLM settings, and path settings
* Falls back to default values if `config.toml` doesn't exist
* Default LLM: `qwen2.5:3b` on `http://localhost:11434` with `4096` token context window

### `src/document_loader.py`

* Uses `docling` to load and convert files from `docs/`
* Converts each document to Markdown
* Adds metadata fields: `source`, `filename`, and `directory`

### `src/text_splitter.py`

* Splits documents into chunks using `CharacterTextSplitter`
* Default chunk size: `500`
* Default overlap: `0`

### `src/vector_store.py`

* Builds a Chroma vector store backed by local embeddings
* Uses `HuggingFaceEmbeddings` and Chroma persistence

### `src/retriever.py`

* Performs similarity search against the vector store
* Returns top-k matches for use in the RAG chain

### `src/llm.py`

* Initializes a `ChatOllama` instance for local LLM inference
* Ensures the requested model is downloaded from Ollama (with user consent)
* Handles interactive model downloading with progress tracking
* Provides error handling for Ollama connection and API issues

## Repository structure

```text
rag-system/
├── docs/                          # Raw documents for ingestion
├── db/                            # Persistent Chroma DB storage
├── src/                           # Core application modules
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py                     # Local LLM initialization and management
│   └── pipelines/                 # Pipeline implementations
│       ├── __init__.py
│       ├── base.py                # Abstract base pipeline class
│       ├── ingestion_pipeline.py  # Document ingestion pipeline
│       ├── retrieval_pipeline.py  # Interactive retrieval pipeline
│       └── scraping_pipeline.py   # Web scraping pipeline
├── main.py                        # CLI entry point
├── config.toml                    # Configuration file
└── requirements.txt               # Python dependencies
```

## Notes

* Configuration is managed via `config.toml` (auto-generated with defaults if missing).
* **Ollama must be running** before starting a retrieval session. Start it with `ollama serve`.
* To use CPU instead of GPU for embeddings, edit `config.toml` and change `device = "cuda"` to `device = "cpu"`.
* The ingestion pipeline expects at least one valid document in the configured `docs_path`.
* Pass `with_previews=True` to pipeline instances if you want sample output during ingestion.
* For web scraping functionality, set the `SERPAPI_KEY` environment variable with your SerpAPI API key.
* Scraped documents are saved as JSON files in the `docs/` directory and can be ingested with the standard ingestion pipeline.
* The `ScrapingPipeline` uses Google as the default search engine. To use a different engine, instantiate `ScrapingPipeline(docs_path=config.DOCS_PATH, engine="bing")` (or any other SerpAPI-supported engine).
* If the requested LLM model is not found locally, the application will prompt you to download it from Ollama.
* The RAG chain uses a system prompt to ensure the LLM only answers from provided context, reducing hallucinations.


