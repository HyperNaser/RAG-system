import json
import io

from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from docling.datamodel.base_models import ConversionStatus

def load_documents(docs_directory: str, with_preview: bool = False) -> list[Document]:
    """Loads all documents in docs directory"""
    docs_path = Path(docs_directory)
    if not docs_path.exists():
        raise FileNotFoundError(f"The directory '{docs_path}' does not exists. Please create it and fill it with documents.")

    metadata_registry: dict[str, Any] = {}

    def file_stream_generator():
        for root, _, files in docs_path.walk():
            for file in files:
                if file.startswith('.'):
                    continue

                path = root / file

                if path.suffix.lower() == ".json":
                    try:
                        with open(path, "r", encoding="utf-8") as json_file:
                            data = json.load(json_file)

                        raw_html = data.get("raw_html", None)
                        if not raw_html:
                            continue

                        stream_key = f"{path}.html"

                        metadata_registry[stream_key] = {
                            "link": data.get("link", ""),
                            "title": data.get("title", ""),
                            **data.get("metadata", {})
                        }

                        yield DocumentStream(
                            name=stream_key,
                            stream=io.BytesIO(raw_html.encode("utf-8")),
                        )
                    except Exception as e:
                        print(f"Skipping malformed JSON wrapper '{file}': {e}")
                else:
                    yield path

    converter = DocumentConverter()
            
    results = converter.convert_all(file_stream_generator())

    documents: list[Document] = []
    
    for result in results:
        if result.status == ConversionStatus.SUCCESS:
            source = str(result.input.file)
            markdown_content = result.document.export_to_markdown()
            
            if source in metadata_registry:
                raw_metadata = metadata_registry[source]
            else: 
                file_path = Path(source)
                raw_metadata = {
                    "source": source,
                    "filename": file_path.name,
                    "directory": str(file_path.parent)
                }
            
            metadata = {}
            
            for key, value in raw_metadata.items():
                if isinstance(value, (dict, list)):
                    metadata[key] = json.dumps(value) 
                else:
                    metadata[key] = value

            doc = Document(
                page_content=markdown_content,
                metadata=metadata
            )
            documents.append(doc)
        else:
            print(f"Failed to convert file '{result.input.file}' with status '{result.status.name}'")
            
    if len(documents) == 0:
        raise FileNotFoundError(f"No files found in '{docs_path}'. Please add documents.")
    
    if with_preview:
        print("Preview of first three documents:")
        for i, doc in enumerate(documents[0:3]):
            print(f"\tDocument {i+1}:")
            print(f"\t\tsource: {doc.metadata["source"]}")
            print(f"\t\tLength: {len(doc.page_content)}")
            print(f"\t\tcontent: {doc.page_content[:50]}...")
        
    return documents