#type: ignore
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.settings import settings
from docling.datamodel.base_models import ConversionStatus
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def _get_files(directory: Path):
    """Creates a file path generator"""
    for root, dirs, files in directory.walk():
        for file in files:
            if file.startswith('.'):
                continue

            path = root / file
            yield path


def load_documents(docs_directory: str, with_preview: bool = False) -> list[Document]:
    """Loads all documents in docs directory"""
    docs_path = Path(docs_directory)
    if not docs_path.exists:
        raise FileNotFoundError(f"The directory '{docs_path}' does not exists. Please create it and fill it with documents.")

    converter = DocumentConverter()

    file_stream = _get_files(docs_path)
            
    results = converter.convert_all(file_stream)

    documents: list[Document] = []
    
    for result in results:
        if result.status == ConversionStatus.SUCCESS:
            file_path = Path(result.input.file)
            markdown_content = result.document.export_to_markdown()
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "directory": str(file_path.parent)
                }
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

def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 0, with_preview: bool = False):
    """Splits documents into chunks with overlap"""
    
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    
    if with_preview:
        print("Preview of first 3 chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\tChunk {i+1}:")
            print(f"\t\tSource: {chunk.metadata["source"]}")
            print(f"\t\tLength: {len(chunk.page_content)}")
            print(f"\t\tContent: {chunk.page_content[:50]}...")
    
    return chunks

def main():
    # Load the files
    documents = load_documents("docs", with_preview=True)

    # chunk the files
    chunks = split_documents(documents, with_preview=True)
    
    # embedding the chunks
    # store the embeddings

if __name__ == "__main__":
    load_dotenv()
    main()