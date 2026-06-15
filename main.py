
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — makes `src/` importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestor import PDFIngestor
from vector_store import PolicyVectorStore
from chain import PolicyChain


# Constants
DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma_db")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

QUICK_QUESTIONS = [
    "What is the waiting period for pre-existing diseases?",
    "Is there a co-pay clause? What percentage?",
    "What are the sub-limits for room rent?",
    "What procedures are excluded from coverage?",
    "What is the sum insured and how can it be enhanced?",
    "What is the grace period for renewal?",
]

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Policy Interrogator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main background */
        .stApp { background-color: #0f1117; }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: #161b27; }

        /* Chat bubbles */
        .user-bubble {
            background: #1e3a5f;
            border-radius: 12px 12px 2px 12px;
            padding: 10px 16px;
            margin: 6px 0 6px 60px;
            color: #e2e8f0;
            font-size: 0.95rem;
        }
        .assistant-bubble {
            background: #1a2235;
            border: 1px solid #2d3748;
            border-radius: 12px 12px 12px 2px;
            padding: 14px 18px;
            margin: 6px 60px 6px 0;
            color: #e2e8f0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .source-tag {
            display: inline-block;
            background: #2d3748;
            color: #90cdf4;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.78rem;
            margin: 3px 3px 0 0;
        }
        .quick-pill {
            font-size: 0.82rem;
        }
        .stSpinner > div { color: #63b3ed; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages": [],          # {"role": "user"|"assistant", "content": str, "sources": list}
        "vector_store": None,
        "policy_chain": None,
        "indexed_files": [],
        "active_source": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# Backend helpers (cached)
@st.cache_resource(show_spinner="Connecting to vector store …")
def get_vector_store() -> PolicyVectorStore:
    return PolicyVectorStore(persist_dir=CHROMA_DIR)


def get_chain(source_filter: str | None = None) -> PolicyChain:
    """Build / rebuild the chain, keyed by active source."""
    store = get_vector_store()
    return PolicyChain(
        vector_store=store,
        model="mistral",
        temperature=0.1,
        k_docs=6,
        source_filter=source_filter,
    )



# Sidebar
with st.sidebar:
    st.markdown("##  Policy Interrogator")
    st.caption("RAG-powered insurance policy analysis")
    st.divider()

    # --- Upload ---
    st.markdown("### Upload Policy PDF")
    uploaded = st.file_uploader(
        label="Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        save_path = DATA_DIR / uploaded.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            st.success(f"Saved: {uploaded.name}")

        if st.button("Index this PDF", use_container_width=True):
            with st.spinner("Parsing and embedding — this takes ~30 s per 60 pages …"):
                ingestor = PDFIngestor(chunk_size=600, overlap=120)
                chunks = ingestor.ingest(save_path)
                store = get_vector_store()
                added = store.add_chunks(chunks)
            st.success(f"Added {added} new chunks to the store.")
            if uploaded.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(uploaded.name)

    st.divider()

    # --- Source selector ---
    st.markdown("### Active Policy")
    store = get_vector_store()
    sources = store.list_sources()

    if sources:
        choice = st.selectbox(
            "Filter responses to:",
            options=["All indexed policies"] + sources,
            index=0,
        )
        active = None if choice == "All indexed policies" else choice
        if active != st.session_state.active_source:
            st.session_state.active_source = active
            st.session_state.policy_chain = None   # force rebuild
            st.session_state.messages = []
    else:
        st.info("No policies indexed yet. Upload a PDF above.")

    st.divider()

    # --- Stats ---
    st.markdown("### Store Stats")
    total_chunks = store.count()
    st.metric("Total chunks", total_chunks)
    st.metric("Policies indexed", len(sources))

    if st.button("🗑️ Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.policy_chain = None
        st.rerun()



# Main area
st.markdown("## Insurance Policy Q&A")

active_source = st.session_state.active_source
if active_source:
    st.caption(f"Querying: **{active_source}**")
else:
    st.caption("Querying: all indexed policies")

# Lazy-build chain
if st.session_state.policy_chain is None and store.count() > 0:
    with st.spinner("Loading Mistral via Ollama …"):
        st.session_state.policy_chain = get_chain(source_filter=active_source)

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            cols_per_row = 4
            source_html = ""
            for s in msg["sources"]:
                label = f"📄 {s['source']}  p.{s['page']}"
                source_html += f'<span class="source-tag">{label}</span>'
            st.markdown(source_html, unsafe_allow_html=True)

# Quick questions
if not st.session_state.messages and store.count() > 0:
    st.markdown("#### Try a quick question:")
    cols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        if cols[i % 2].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state._quick_prompt = q
            st.rerun()

# Process quick prompt 
if hasattr(st.session_state, "_quick_prompt") and st.session_state._quick_prompt:
    prompt = st.session_state._quick_prompt
    del st.session_state._quick_prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Retrieving from policy …"):
        result = st.session_state.policy_chain.ask(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    st.rerun()

# Text input
if store.count() == 0:
    st.warning("Index at least one PDF using the sidebar before asking questions.")
else:
    user_input = st.chat_input("Ask anything about the policy …")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking …"):
            result = st.session_state.policy_chain.ask(user_input)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )
        st.rerun()