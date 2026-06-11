#type: ignore
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def load_documents(docs_directory: str, with_preview: bool = False) -> list[Document]:
    docs_path = Path(docs_directory)
    if not docs_path.exists:
        raise FileNotFoundError(f"The directory '{docs_path}' does not exists. Please create it and fill it with documents.")

    converter = DocumentConverter()

    documents: list[Document] = []
    for root, dirs, files in docs_path.walk():
        for file in files:
            if file.startswith('.'):
                continue

            path = root / file
            
            try:
                result = converter.convert(str(path))
                markdown_content = result.document.export_to_markdown()
                
                doc = Document(
                    page_content=markdown_content,
                    metadata={
                        "source": str(path),
                        "filename": file,
                        "directory": str(root)
                    }
                )
                documents.append(doc)

            except Exception as e:
                print(f"Skipping {file} due to an error: {e}")

    if len(documents) == 0:
        raise FileNotFoundError(f"No files found in '{docs_path}'. Please add documents.")
    
    if with_preview:
        print("Preview of first three documents:")
        for i, doc in enumerate(documents[0:3]):
            print(f"\tDocument {i+1}:")
            print(f"\t\tsource: {doc.metadata["source"]}")
            print(f"\t\tcontent: {doc.page_content[0:50]}...")
        
    return documents
    
def main():
    # Load the files
    documents = load_documents("docs", with_preview=True)

    # chunk the files
    # embedding the chunks
    # store the embeddings

if __name__ == "__main__":
    load_dotenv()
    main()