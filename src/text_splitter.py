from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 0, with_preview: bool = False) -> list[Document]:
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