# Local RAG Experimentation Platform

A modular, local Retrieval-Augmented Generation (RAG) backend built using Python. This project is structured to parse local documents, split them into chunks, generate embeddings locally, and manage a persistent vector database. It includes a command-line interface supporting both database ingestion and an interactive multi-query retrieval session.

## Architecture and Design

The project uses a modular Controller/Pipeline pattern to separate user interfaces, orchestration logic, and independent functional modules.

* **Entry Interface (`main.py`)**: A routing script that uses command-line arguments to determine the execution mode. It contains no implementation logic.
* **Pipeline Controller (`src/pipelines.py`)**: Binds structural workflows together, coordinating data movement from ingestion to storage, or initiating interactive retrieval sessions.
* **Functional Modules**: Independent utilities handling specific tasks (document loading, text splitting, storage management, and vector retrieval).

```text
RAG-system/
│
├── docs/                      # Directory for raw documents (PDF, MD, DOCX, etc.)
├── db/                        # Persistent storage for ChromaDB data
│
├── src/                       # Core application package
│   ├── __init__.py            # Packages the src directory
│   ├── document_loader.py     # Document indexing and parsing via Docling
│   ├── text_splitter.py       # Document chunking and overlap configurations
│   ├── vector_store.py        # Embedding model configuration and Chroma control
│   ├── retriever.py           # Similarity search logic
│   └── pipelines.py           # Workflow orchestration (Ingestion vs. Query Loop)
│
├── main.py                    # Application entry point router
└── requirements.txt           # Python project dependencies