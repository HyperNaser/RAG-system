from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .base import BasePipeline
from src.config import AppConfig
from src.llm import get_local_llm

class RetrievalPipeline(BasePipeline):
    def __init__(self, vector_store: VectorStore, app_config: AppConfig) -> None:
        self.vector_store = vector_store
        self.config = app_config

        self.llm = get_local_llm(self.config)
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.rag_chain = self._build_rag_chain()
        
    def _format_docs(self, docs: list[Document]):
        """Converts retrieved Document chunks into a structured text wall."""
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('filename', 'Unknown')}]\n{doc.page_content}" 
            for doc in docs
        )
    
    def _build_rag_chain(self):
        template = """You are a focused assistant. Answer the user question based ONLY on the provided text context. 
If the text does not contain the answer, explicitly state: "I am sorry, I do not have enough information to answer that." 
Do not invent or extrapolate details outside of the provided context.

Context:
{context}

Question: {question}
Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # pipeline: Retrieval -> Context Formatting -> Prompt -> LLM -> String Output
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _execute(self):
        print("\nInteractive RAG Session Started! (Type 'exit', 'q' or 'quit' to stop)")
        print("=" * 60)
        
        while True:
            try:
                user_query = input("\nEnter your search query: ").strip()
                
                if user_query.lower() in ["exit", "quit", "q"]:
                    print("\nClosing interactive retrieval session. Goodbye!")
                    break
                    
                if not user_query:
                    continue
                    
                print("Retrieving context and generating response...")

                response = self.rag_chain.invoke(user_query)
                
                print("\nResponse:")
                print(response)
                print("=" * 60)
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
