#type: ignore
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def main():
    # Load the files
    # chunk the files
    # embedding the chunks
    # store the embeddings
    print("from main")

if __name__ == "__main__":
    load_dotenv()
    main()