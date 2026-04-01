"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager, get_default_vector_store


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Same process singleton as LangGraph retrieval (shared Chroma + embeddings)."""
    return get_default_vector_store()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="stApp"] {
    font-family: 'Outfit', system-ui, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(99, 102, 241, 0.35) 0%, transparent 50%),
                radial-gradient(ellipse 80% 60% at 100% 50%, rgba(236, 72, 153, 0.12) 0%, transparent 45%),
                linear-gradient(168deg, #0c0c12 0%, #12101c 40%, #0d1526 100%);
    color: #e6e8ef;
}

[data-testid="stHeader"] { background: rgba(10, 10, 18, 0.85); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(255,255,255,0.06); }

h1 { font-weight: 600; letter-spacing: -0.03em; background: linear-gradient(120deg, #a5b4fc 0%, #f0abfc 50%, #67e8f9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(18, 16, 28, 0.97) 0%, rgba(12, 18, 32, 0.98) 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] .stMarkdown h3 { color: #a5b4fc !important; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em; }

.block-container { padding-top: 2rem; max-width: 1600px; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 4px; }

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(236, 72, 153, 0.25)) !important;
    color: #f1f5f9 !important;
}

.stChatMessage[data-testid="stChatMessage-user"] { background: rgba(99, 102, 241, 0.12); border-radius: 14px; border: 1px solid rgba(99, 102, 241, 0.2); }
.stChatMessage[data-testid="stChatMessage-assistant"] { background: rgba(15, 23, 42, 0.55); border-radius: 14px; border: 1px solid rgba(148, 163, 184, 0.12); }

.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    border: none;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}
.stButton > button:hover { filter: brightness(1.08); box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4); }

div[data-baseweb="select"] > div { border-radius: 10px !important; border-color: rgba(255,255,255,0.15) !important; background: rgba(0,0,0,0.25) !important; }

.stMetric { background: rgba(255,255,255,0.04); padding: 0.75rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); }
.stMetric label { color: #94a3b8 !important; }
.stMetric [data-testid="stMetricValue"] { color: #e0e7ff !important; }

.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 6px;
    background: rgba(99, 102, 241, 0.25);
    color: #c7d2fe;
    border: 1px solid rgba(129, 140, 248, 0.35);
}
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


TOPIC_OPTIONS = [
    "All topics",
    "ANN",
    "CNN",
    "RNN",
    "LSTM",
    "Seq2Seq",
    "Autoencoder",
    "GAN",
    "SOM",
    "BoltzmannMachine",
    "General",
]

DIFFICULTY_OPTIONS = ["All levels", "beginner", "intermediate", "advanced"]


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.markdown("### Corpus")
    st.sidebar.caption("Ingest PDFs and Markdown into ChromaDB")

    uploaded = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        help="Files should follow topic_difficulty.md when possible.",
    )

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        ingest_btn = st.button("Ingest", type="primary", use_container_width=True)
    with col_b:
        if st.button("Refresh list", use_container_width=True):
            st.session_state["ingested_documents"] = store.list_documents()
            st.rerun()

    if ingest_btn and uploaded:
        with tempfile.TemporaryDirectory(prefix="rag_upload_") as td:
            paths: list[Path] = []
            for uf in uploaded:
                dest = Path(td) / uf.name
                dest.write_bytes(uf.getvalue())
                paths.append(dest)
            chunks = chunker.chunk_files(paths)
            if not chunks:
                st.sidebar.warning("No chunks produced — check file encoding and content.")
            else:
                result = store.ingest(chunks)
                st.session_state["last_ingestion_result"] = result
                st.session_state["ingested_documents"] = store.list_documents()
                if result.ingested:
                    st.sidebar.success(
                        f"Added **{result.ingested}** new chunks · "
                        f"skipped **{result.skipped}** duplicates"
                    )
                elif result.skipped:
                    st.sidebar.info(
                        f"All **{result.skipped}** chunks were already in the store."
                    )
                for err in result.errors:
                    st.sidebar.error(err)

    elif ingest_btn and not uploaded:
        st.sidebar.warning("Select one or more files first.")

    docs = st.session_state.get("ingested_documents") or store.list_documents()
    st.session_state["ingested_documents"] = docs

    if docs:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Indexed documents**")
        for d in docs:
            row1, row2 = st.sidebar.columns([3, 1])
            with row1:
                st.caption(f"{d['source']} · {d['chunk_count']} chunks")
            with row2:
                if st.button("✕", key=f"del_{d['source']}", help="Remove from index"):
                    store.delete_document(d["source"])
                    st.session_state["ingested_documents"] = store.list_documents()
                    if st.session_state.get("selected_document") == d["source"]:
                        st.session_state["selected_document"] = None
                    st.rerun()
    else:
        st.sidebar.info("No documents yet — upload `.md` or `.pdf` to begin.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    try:
        stats = store.get_collection_stats()
    except Exception:
        return
    if stats["total_chunks"] == 0:
        return
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.metric("Chunks", stats["total_chunks"])
    with c2:
        st.metric("Topics", len(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus corpus material detected")
    topics_preview = ", ".join(stats["topics"][:6])
    if len(stats["topics"]) > 6:
        topics_preview += "…"
    st.sidebar.caption(topics_preview or "—")


def render_document_viewer(store: VectorStoreManager) -> None:
    st.markdown("#### Document viewer")
    docs = store.list_documents()
    if not docs:
        st.markdown(
            '<p style="color:#94a3b8;font-size:1rem;">'
            "Ingest documents from the sidebar to browse chunks and metadata here."
            "</p>",
            unsafe_allow_html=True,
        )
        return

    options = [d["source"] for d in docs]
    idx = 0
    current = st.session_state.get("selected_document")
    if current in options:
        idx = options.index(current)

    choice = st.selectbox(
        "Select document",
        options,
        index=idx,
        label_visibility="collapsed",
    )
    st.session_state["selected_document"] = choice

    chunks = store.get_document_chunks(choice)
    st.caption(f"{len(chunks)} chunks · {choice}")

    body = st.container(height=420)
    with body:
        for i, ch in enumerate(chunks, start=1):
            meta = ch.metadata
            st.markdown(
                f'<span class="badge">{meta.topic}</span>'
                f'<span class="badge">{meta.difficulty}</span>'
                f'<span class="badge">{meta.type}</span>',
                unsafe_allow_html=True,
            )
            if meta.is_bonus:
                st.markdown('<span class="badge">BONUS</span>', unsafe_allow_html=True)
            st.markdown(ch.chunk_text)
            if i < len(chunks):
                st.divider()


def render_chat_interface(graph) -> None:
    st.markdown("#### Interview prep chat")

    c1, c2, c3 = st.columns([1.1, 1.1, 1])
    with c1:
        topic_choice = st.selectbox(
            "Topic filter",
            TOPIC_OPTIONS,
            index=0,
        )
    with c2:
        diff_choice = st.selectbox(
            "Difficulty",
            DIFFICULTY_OPTIONS,
            index=0,
        )
    with c3:
        st.write("")
        st.write("")
        if st.button("New conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()

    topic_f = None if topic_choice == "All topics" else topic_choice
    diff_f = None if diff_choice == "All levels" else diff_choice

    chat_box = st.container(height=400)
    with chat_box:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("No relevant content matched the corpus — hallucination guard active.")
                if message.get("rewritten_query"):
                    with st.expander("Retrieval query used"):
                        st.code(message["rewritten_query"], language=None)

    query = st.chat_input("Ask about neural networks, CNNs, RNNs, training…")

    if query is not None:
        q = query.strip()
        if not q:
            st.warning("Enter a question to send.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": q})
            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=q)],
                        "topic_filter": topic_f,
                        "difficulty_filter": diff_f,
                    },
                    config={
                        "configurable": {"thread_id": st.session_state.thread_id}
                    },
                )
                fr: AgentResponse | None = result.get("final_response")
                if fr:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": fr.answer,
                            "sources": fr.sources,
                            "no_context_found": fr.no_context_found,
                            "rewritten_query": fr.rewritten_query,
                        }
                    )
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "No response object returned from the agent.",
                            "sources": [],
                            "no_context_found": True,
                            "rewritten_query": "",
                        }
                    )
            except Exception as e:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"**Error:** {e}",
                        "sources": [],
                        "no_context_found": False,
                        "rewritten_query": "",
                    }
                )
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_theme()

    st.markdown(f"# 🧠 {settings.app_title}")
    st.caption(
        "Grounded RAG with LangGraph · duplicate-aware ingestion · citation-first answers"
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    v_col, c_col = st.columns([1, 1], gap="large")

    with v_col:
        render_document_viewer(store)

    with c_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
