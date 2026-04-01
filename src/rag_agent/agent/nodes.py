"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from rag_agent.agent.prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import get_default_vector_store


def _last_human_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    original = _last_human_text(state).strip()
    if not original:
        return {"original_query": "", "rewritten_query": ""}

    llm = LLMFactory().create()
    prompt = QUERY_REWRITE_PROMPT.format(original_query=original)
    try:
        out = llm.invoke(prompt)
        text = getattr(out, "content", str(out))
        rewritten = text.strip() if isinstance(text, str) else str(text).strip()
    except Exception:
        rewritten = original

    if not rewritten:
        rewritten = original
    return {"original_query": original, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = get_default_vector_store()
    q = (state.get("rewritten_query") or state.get("original_query") or "").strip()
    if not q:
        return {"retrieved_chunks": [], "no_context_found": True}

    chunks: list[RetrievedChunk] = manager.query(
        q,
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter"),
    )
    if not chunks:
        return {"retrieved_chunks": [], "no_context_found": True}
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard when no_context_found is True.

    Parameters
    ----------
    state : AgentState
        Current graph state.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    if state.get("no_context_found"):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=str(state.get("rewritten_query") or ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    context = "\n\n".join(
        f"{c.to_citation()}\n{c.chunk_text}" for c in chunks
    )
    avg_confidence = (
        sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
    )
    sources = [c.to_citation() for c in chunks]
    original_q = str(state.get("original_query") or _last_human_text(state))

    prior = list(state.get("messages") or [])
    context_tokens = len(tiktoken.get_encoding("cl100k_base").encode(context))
    history_budget = max(
        400,
        settings.max_context_tokens - context_tokens - 400,
    )
    trimmed = trim_messages(
        prior,
        max_tokens=history_budget,
        strategy="last",
        token_counter=count_tokens_approximately,
        start_on="human",
        include_system=False,
    )

    user_turn = HumanMessage(
        content=(
            "Use ONLY the retrieved context below. Cite sources as "
            "[SOURCE: topic | filename] for every factual claim.\n\n"
            f"{context}\n\nUser question:\n{original_q}"
        )
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed + [user_turn]

    try:
        ai_msg = llm.invoke(messages)
        answer_text = ai_msg.content if isinstance(ai_msg.content, str) else str(
            ai_msg.content
        )
    except Exception as e:
        answer_text = (
            f"The model failed to generate a response ({e!s}). "
            "Check your LLM provider configuration in .env."
        )
        ai_msg = AIMessage(content=answer_text)

    response = AgentResponse(
        answer=answer_text,
        sources=sources,
        confidence=float(avg_confidence),
        no_context_found=False,
        rewritten_query=str(state.get("rewritten_query") or ""),
    )
    return {
        "final_response": response,
        "messages": [ai_msg if isinstance(ai_msg, AIMessage) else AIMessage(content=answer_text)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Route after retrieval. Always continue to generation so the
    hallucination guard and user-facing message stay in one place.
    """
    return "generate"
