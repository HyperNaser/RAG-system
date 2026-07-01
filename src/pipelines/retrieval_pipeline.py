from typing import Any

from langchain.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores.base import VectorStore

from src.config import AppConfig
from src.llm import get_local_llm

from .base import BasePipeline


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
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Given a chat history and the latest user question which might reference context in the chat history, "
                        "formulate a standalone question which can be understood without the chat history. "
                        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        condense_chain = contextualize_prompt | self.llm | StrOutputParser()

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a focused assistant. Answer the user question based ONLY on the provided text context.\n"
                        "If the text does not contain the answer, explicitly state: 'I am sorry, I do not have enough information to answer that.'\n"
                        "Do not invent or extrapolate details outside of the provided context.\n\n"
                        "Context:\n{context}"
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def _get_context_and_docs(inputs: dict[str, Any]):
            if inputs.get("chat_history"):
                standalone_question = condense_chain.invoke(inputs)
            else:
                standalone_question = inputs["question"]

            docs = self.retriever.invoke(standalone_question)
            return self._format_docs(docs)

        return (
            RunnablePassthrough.assign(context=_get_context_and_docs)
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def _execute(self):
        print("\nInteractive RAG Session Started! (Type 'exit', 'q' or 'quit' to stop)")
        print("=" * 60)

        chat_history: list[BaseMessage] = []

        while True:
            try:
                user_query = input("\nEnter your search query: ").strip()

                if user_query.lower() in ["exit", "quit", "q"]:
                    print("\nClosing interactive retrieval session. Goodbye!")
                    break

                if not user_query:
                    continue

                print("Retrieving context and generating response...")

                response_chunks: list[str] = []

                stream_generator = self.rag_chain.stream(
                    {"question": user_query, "chat_history": chat_history}
                )

                for chunk in stream_generator:
                    print(chunk, end="", flush=True)
                    response_chunks.append(chunk)

                print("\n\n" + "=" * 60)

                if response_chunks:
                    full_response = "".join(response_chunks)
                    chat_history.append(HumanMessage(content=user_query))
                    chat_history.append(AIMessage(content=full_response))
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
