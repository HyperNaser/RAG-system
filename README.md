# Local RAG Experimentation Platform

A modular, local Retrieval-Augmented Generation (RAG) backend built in Python. This repository ingests local documents, converts them to Markdown, splits text into retrievable chunks, generates local embeddings, and stores them in a persistent Chroma vector database.

## Features

* Local document ingestion using `docling`
* Markdown conversion and metadata extraction
* Chunking with `langchain_text_splitters` (`chunk_size=500`, `chunk_overlap=0`)
* Local embedding generation with `HuggingFaceEmbeddings`
* Persistent vector storage with `Chroma`
* Interactive CLI retrieval session with top-k similarity search

## Prerequisites

* Python environment with dependencies installed from `requirements.txt`
* A `docs/` folder containing the source documents to ingest
* A local GPU if you want to use the default `device="cuda"` setting
* `db/chroma_db` is used to persist the vector store between runs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Add your documents to the `docs/` directory.
2. Build the vector store:

```bash
python main.py --ingest
```

3. Start the interactive retrieval session:

```bash
python main.py --query
```

4. In query mode, type a question and press Enter. Type `exit`, `quit`, or `q` to stop.

## Configuration

The project can be configured via `config.toml`:

```toml
[embedding]
model_name = "BAAI/bge-small-en-v1.5"
device = "cuda"

[paths]
db_path = "db/chroma_db"
docs_path = "docs"
```

If `config.toml` is missing, the application uses default values (as shown above). You can override any setting by editing the config file.

## How it works

### `main.py`

* Loads configuration from `config.toml` (or uses defaults)
* Instantiates a Hugging Face embedding model based on config
* Creates a persistent Chroma vector store
* Runs either the ingestion pipeline or the retrieval pipeline based on CLI flags

### `src/pipelines.py`

* `run_ingestion_pipeline()`
  * Loads documents from `docs/`
  * Splits them into chunks
  * Optionally resets the vector store when `overwrite=True`
  * Adds chunks to the Chroma database
* `run_retrieval_pipeline()`
  * Starts an interactive loop
  * Queries the vector store for the top 3 most relevant chunks
  * Prints retrieved chunk previews and source metadata

### `src/config.py`

* Loads configuration from `config.toml`
* Provides an `AppConfig` dataclass with model name, device, and path settings
* Falls back to default values if `config.toml` doesn't exist

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
* Returns top-k matches and prints the first 200 characters of each match

## Repository structure

```text
RAG-system/
├── docs/                      # Raw documents for ingestion
├── db/                        # Persistent Chroma DB storage
├── src/                       # Core application modules
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── pipelines.py
├── main.py                    # CLI entry point
├── config.toml                # Configuration file
└── requirements.txt           # Python dependencies
```

## Notes

* Configuration is managed via `config.toml` (auto-generated with defaults if missing).
* To use CPU instead of GPU, edit `config.toml` and change `device = "cuda"` to `device = "cpu"`.
* The ingestion pipeline expects at least one valid document in the configured `docs_path`.
* Pass `with_preview=True` to the pipeline functions if you want sample output during ingestion.
