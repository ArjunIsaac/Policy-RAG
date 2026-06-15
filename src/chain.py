"""
chain.py
--------
Builds the LangChain retrieval-augmented generation (RAG) chain.

LLM    : Mistral via Ollama (local, no API key)
Memory : ConversationBufferWindowMemory  (last N turns kept in context)
Prompt : Strict compliance-oriented system prompt tuned for insurance Q&A
"""

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama

from vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are an expert insurance policy analyst assistant helping \
financial advisors quickly understand complex policy documents.

STRICT RULES you must always follow:
1. Base every answer EXCLUSIVELY on the retrieved context passages provided below.
2. If the context does not contain enough information, say explicitly:
   "I could not find sufficient information in the provided policy document."
   Do NOT guess or hallucinate figures.
3. When quoting waiting periods, co-pay percentages, sub-limits, or exclusions,
   always cite the page number(s) from the metadata.
4. Present numerical values (amounts, percentages, days) in a clearly labelled
   format, e.g. "Waiting Period: 30 days (Page 12)".
5. If a question is ambiguous, ask ONE clarifying question before answering.
6. Keep responses concise but complete. Use bullet points for lists of clauses.

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}
"""

HUMAN_TEMPLATE = "{question}"


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ]
    )


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def build_chain(
    vector_store: PolicyVectorStore,
    model: str = "mistral",
    temperature: float = 0.1,
    k_docs: int = 6,
    memory_window: int = 6,
    source_filter: str | None = None,
) -> ConversationalRetrievalChain:
    """
    Construct and return a ConversationalRetrievalChain.

    Parameters
    ----------
    vector_store   : initialised PolicyVectorStore
    model          : Ollama model name (default: "mistral")
    temperature    : LLM temperature; keep low for factual extraction
    k_docs         : number of context chunks to retrieve per query
    memory_window  : how many past turns to keep in context
    source_filter  : restrict retrieval to one PDF filename (optional)

    Returns
    -------
    ConversationalRetrievalChain ready for `.invoke()`
    """
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        # Request longer outputs so multi-clause answers aren't truncated
        num_predict=1024,
    )

    retriever = vector_store.as_retriever(k=k_docs, source_filter=source_filter)

    memory = ConversationBufferWindowMemory(
        k=memory_window,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": _build_prompt()},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )

    return chain


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class PolicyChain:
    """
    Thin wrapper around ConversationalRetrievalChain that exposes a clean
    `.ask(question)` API and formats source citations automatically.
    """

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        model: str = "mistral",
        temperature: float = 0.1,
        k_docs: int = 6,
        memory_window: int = 6,
        source_filter: str | None = None,
    ) -> None:
        self._chain = build_chain(
            vector_store=vector_store,
            model=model,
            temperature=temperature,
            k_docs=k_docs,
            memory_window=memory_window,
            source_filter=source_filter,
        )

    def ask(self, question: str) -> dict:
        """
        Send a question to the chain.

        Returns
        -------
        dict with keys:
            answer   : str   — LLM response
            sources  : list  — [{"source": filename, "page": N, "text": snippet}]
        """
        result = self._chain.invoke({"question": question})

        sources = []
        seen: set[str] = set()
        for doc in result.get("source_documents", []):
            key = f"{doc.metadata.get('source')}::{doc.metadata.get('page')}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": doc.metadata.get("source", ""),
                        "page": doc.metadata.get("page", "?"),
                        "snippet": doc.page_content[:200].replace("\n", " "),
                    }
                )

        return {"answer": result["answer"], "sources": sources}

    def reset_memory(self) -> None:
        """Clear the conversation history."""
        self._chain.memory.clear()