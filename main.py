"""
main.py — Policy Interrogator
Streamlit UI with:
  • Multi-document upload & indexing
  • Per-policy attribute extraction dashboard
  • Side-by-side policy comparison
  • Conversation loop with page + line citations
  • Debug retrieval tool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote
from langchain_openai import ChatOpenAI

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from constants import MODEL_NAME

from ingestor import PDFIngestor
from vector_store import PolicyVectorStore
from policy_chain import PolicyChain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NOTE: PDFs now live under ./static/raw so Streamlit's static file server
# (enabled in .streamlit/config.toml) can serve them directly — this is what
# lets the page-number citations link straight to the right page of the PDF.
DATA_DIR = Path("static/raw")
CHROMA_DIR = Path("data/chroma_db")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

QUICK_QUESTIONS = [
    "What is the waiting period for pre-existing diseases?",
    "Is there a co-pay clause? What percentage applies?",
    "What are the sub-limits for room rent and ICU?",
    "List all permanent exclusions.",
    "What is the No Claim Bonus benefit?",
    "What are the sum insured options?",
    "How many network hospitals are covered?",
    "What is the grace period for renewal?",
]

ATTR_LABELS: dict[str, str] = {
    "policy_name": "Policy Name",
    "insurer": "Insurer",
    "sum_insured_options": "Sum Insured Options",
    "policy_tenure": "Policy Tenure",
    "lifetime_renewability": "Lifetime Renewable",
    "free_look_period_days": "Free Look Period (days)",
    "grace_period_days": "Grace Period (days)",
    "waiting_period_initial_days": "Initial Waiting Period (days)",
    "waiting_period_ped_months": "PED Waiting Period (months)",
    "waiting_period_specific_illness_months": "Specific Illness Waiting Period (months)",
    "copay_applicable": "Co-pay Applicable",
    "copay_percentage": "Co-pay %",
    "copay_conditions": "Co-pay Conditions",
    "room_rent_sublimit": "Room Rent Sub-limit",
    "icu_sublimit": "ICU Sub-limit",
    "inpatient_covered": "Inpatient Cover",
    "daycare_covered": "Day Care Procedures",
    "domiciliary_covered": "Domiciliary Treatment",
    "maternity_covered": "Maternity Covered",
    "ambulance_covered": "Emergency Ambulance",
    "organ_donor_covered": "Organ Donor Cover",
    "pre_hospitalisation_days": "Pre-Hospitalisation (days)",
    "post_hospitalisation_days": "Post-Hospitalisation (days)",
    "cashless_available": "Cashless Facility",
    "network_hospitals": "Network Hospitals",
    "claim_settlement_days": "Claim Settlement (days)",
    "portability_available": "Portability",
    "ncb_benefit": "No Claim Bonus",
    "permanent_exclusions": "Permanent Exclusions",
}

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Policy Interrogator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  [data-testid="stSidebar"] { background-color: #161b22; }

  .user-bubble {
    background: #1f3a5f; border-radius: 14px 14px 4px 14px;
    padding: 10px 16px; margin: 8px 0 8px 80px; font-size:.95rem; color:#e6edf3;
  }
  .bot-bubble {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px; margin: 8px 80px 8px 0;
    font-size:.95rem; color:#e6edf3; line-height:1.7;
  }
  .cite-badge {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    color:#58a6ff; border-radius:6px; padding:2px 9px;
    font-size:.76rem; margin:3px 3px 0 0; font-family:monospace;
    text-decoration:none; cursor:pointer; transition:background .15s ease;
  }
  .cite-badge:hover { background:#30363d; border-color:#58a6ff; }
  .attr-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:14px 18px; margin-bottom:10px;
  }
  .attr-label { color:#8b949e; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }
  .attr-value { color:#e6edf3; font-size:1rem; font-weight:500; margin-top:2px; }
  .attr-meta { margin-top:6px; display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
  .attr-page {
    display:inline-block; background:#0d419d22; border:1px solid #1f6feb;
    color:#58a6ff; border-radius:4px; padding:1px 7px;
    font-size:.72rem; font-family:monospace; cursor:pointer;
    text-decoration:none; transition:background .15s ease;
  }
  .attr-page:hover { background:#1f6feb44; }
  .attr-clause {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    color:#8b949e; border-radius:4px; padding:1px 7px;
    font-size:.72rem; font-family:monospace;
  }
  .conf-high    { color:#3fb950; font-size:.7rem; }
  .conf-medium  { color:#d29922; font-size:.7rem; }
  .conf-low     { color:#f0883e; font-size:.7rem; }
  .conf-not_found { color:#6e7681; font-size:.7rem; }
  .status-requires { color:#f0883e; font-size:.7rem; font-style:italic; }
  .attr-value-warn { color:#d29922; font-size:.95rem; font-weight:500; margin-top:2px; }
  .attr-value-null { color:#484f58; font-size:.95rem; font-style:italic; margin-top:2px; }
  .section-header {
    color:#58a6ff; font-size:.85rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.1em; margin:18px 0 8px;
  }
  div[data-testid="stTabs"] button { font-size:.9rem; }

  .summary-card {
    background: linear-gradient(135deg, #0d2137 0%, #161b22 100%);
    border: 1px solid #1f6feb; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 20px;
  }
  .summary-title {
    color:#58a6ff; font-size:.8rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.12em; margin-bottom:14px;
  }
  .summary-tier { font-size:1.4rem; font-weight:700; color:#e6edf3; margin-bottom:2px; }
  .summary-conf { color:#3fb950; font-size:.85rem; margin-bottom:14px; }
  .summary-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }
  .summary-item-label { color:#6e7681; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
  .summary-item-value { color:#e6edf3; font-size:.9rem; font-weight:500; margin-top:1px; }
  .conflict-banner { background:#2d1f00; border:1px solid #f0883e; border-radius:8px; padding:10px 14px; margin-top:8px; }
  .conflict-title { color:#f0883e; font-size:.78rem; font-weight:700; margin-bottom:6px; }
  .conflict-val { color:#ffa657; font-size:.88rem; font-weight:600; min-width:80px; display:inline-block; }
  .conflict-clause { color:#8b949e; font-size:.78rem; }
  .evidence-box { background:#0d1117; border:1px solid #21262d; border-radius:6px;
    padding:8px 12px; margin-top:6px; font-size:.78rem;
    color:#8b949e; font-family:monospace; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init():
    defaults = {
        "messages": [],
        "chain": None,
        "selected_sources": [],
        "extracted_attrs": {},   # {source_name: dict}
        "compare_mode": False,
        "debug_output": "",      # store debug result
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to vector store …")
def get_store() -> PolicyVectorStore:
    return PolicyVectorStore(persist_dir=CHROMA_DIR)


def get_chain(sources: list[str] | None) -> PolicyChain:
    return PolicyChain(
        vector_store=get_store(),
        model= MODEL_NAME,  # Pass as string, not ChatOpenAI object
        temperature=0.05,
        k_docs=6,
        source_filter=sources if sources else None,
    )
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_value(v) -> str:
    """Render a value for comparison table — handles citation envelopes and raw values."""
    if isinstance(v, dict) and "value" in v:
        # Citation envelope — use the display string
        return v.get("display") or "Not specified in policy"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if v is None:
        return "Not specified in policy"
    return str(v)


def _pdf_url(source: str, page: int | None = None) -> str:
    """Build the static-served URL for a source PDF, optionally deep-linking to a page.

    Requires the PDF to live under ./static/raw (see DATA_DIR) and
    enableStaticServing = true in .streamlit/config.toml.
    """
    url = f"app/static/raw/{quote(source)}"
    if page:
        url += f"#page={page}"
    return url


def _page_badge_html(source: str | None, page: int | None, css_class: str = "attr-page") -> str:
    """Render a page-number badge. Clickable (opens the PDF at that page) when a
    source filename is known, otherwise a plain non-interactive badge."""
    if not page:
        return ""
    if source:
        return (
            f'<a class="{css_class}" href="{_pdf_url(source, page)}" '
            f'target="_blank" rel="noopener" title="Open PDF at page {page}">📄 p.{page}</a>'
        )
    return f'<span class="{css_class}">p.{page}</span>'


def _attr_card(label: str, value, source: str = "") -> None:
    if isinstance(value, dict) and "value" in value:
        display   = value.get("display") or "Not specified in policy"
        page      = value.get("page")
        clause    = value.get("clause")
        conf      = value.get("confidence", "high")
        status    = value.get("status", "verified")
        raw_val   = value.get("value")
        evidence  = value.get("evidence")
        conflicts = value.get("conflicts", [])

        if raw_val is None:
            val_class = "attr-value-null"
        elif status == "requires_verification":
            val_class = "attr-value-warn"
        else:
            val_class = "attr-value"

        list_display = display
        if isinstance(raw_val, list):
            list_display = "<br>".join(f"• {item}" for item in raw_val)

        conf_icons = {"high": "🟢", "medium": "🟡", "low": "🟠", "not_found": "⚪"}
        conf_icon = conf_icons.get(conf, "⚪")

        page_badge   = _page_badge_html(source, page)
        clause_badge = f'<span class="attr-clause">{clause}</span>' if clause else ""
        status_note  = (
            f'<span class="status-requires">⚠ Requires verification</span>'
            if status == "requires_verification" else ""
        )
        meta_html = ""
        if page_badge or clause_badge or status_note:
            meta_html = (
                f'<div class="attr-meta">'
                f'{page_badge}{clause_badge}'
                f'<span class="conf-{conf}">{conf_icon} {conf}</span>'
                f'{status_note}'
                f'</div>'
            )

        st.markdown(
            f'<div class="attr-card">'
            f'<div class="attr-label">{label}</div>'
            f'<div class="{val_class}">{list_display}</div>'
            f'{meta_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Evidence for the value shown above (collapsed by default)
        if evidence:
            with st.expander("📄 Evidence", expanded=False):
                st.markdown(
                    f'<div class="evidence-box">"{evidence}"</div>',
                    unsafe_allow_html=True,
                )

        # Conflict banner — print EVERY value found across the policy (the one shown
        # above, plus every alternate), each with its own page, clause, and evidence,
        # so nothing is silently buried or dropped.
        if conflicts:
            # Reuse whatever unit text the primary "display" string uses (e.g. "months",
            # "days") so alternates are shown the same way, without hard-coding a unit.
            unit_suffix = ""
            if isinstance(raw_val, (int, float)) and display:
                unit_suffix = display.replace(str(raw_val), "", 1).strip()

            def _fmt_val(v) -> str:
                return f"{v} {unit_suffix}".strip() if unit_suffix else str(v)

            all_candidates = (
                [{"value": raw_val, "page": page, "clause": clause,
                  "evidence": evidence, "is_primary": True}]
                + [dict(c, is_primary=False) for c in conflicts]
            )

            rows_html = ""
            for c in all_candidates:
                tag = (
                    '<span style="color:#3fb950;font-size:.68rem;font-weight:700;">✓ SHOWN ABOVE</span>'
                    if c.get("is_primary") else
                    '<span style="color:#f0883e;font-size:.68rem;font-weight:700;">⚠ ALSO FOUND</span>'
                )
                c_page = _page_badge_html(source, c.get("page"))
                c_evidence = c.get("evidence")
                evidence_html = f'<div class="evidence-box">"{c_evidence}"</div>' if c_evidence else ""
                rows_html += (
                    '<div style="margin:0 0 10px;padding-bottom:8px;border-bottom:1px solid #30363d;">'
                    f'<span class="conflict-val">{_fmt_val(c.get("value"))}</span> '
                    f'<span class="conflict-clause">— {c.get("clause","")}</span> '
                    f'{c_page} {tag}'
                    f'{evidence_html}'
                    '</div>'
                )
            st.markdown(
                '<div class="conflict-banner">'
                '<div class="conflict-title">⚠ Multiple values found across the policy — verify before relying on this field</div>'
                f'{rows_html}'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        # Legacy / raw value
        if isinstance(value, list):
            display = "<br>".join(f"• {item}" for item in value)
        elif isinstance(value, bool):
            display = "Yes" if value else "No"
        elif value is None:
            display = "Not specified in policy"
        else:
            display = str(value)

        val_class = "attr-value-null" if value is None else "attr-value"
        st.markdown(
            f'<div class="attr-card">'
            f'<div class="attr-label">{label}</div>'
            f'<div class="{val_class}">{display}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _summary_card(summary: dict) -> None:
    """Render the top-level policy risk summary card."""
    tier = summary.get("coverage_tier", "—")
    conf = summary.get("overall_confidence", 0)
    tier_color = {"Comprehensive": "#3fb950", "Standard": "#d29922", "Basic": "#f0883e"}.get(tier, "#8b949e")

    conflicts = summary.get("conflicts", [])
    conflict_html = ""
    if conflicts:
        names = ", ".join(c.replace("_", " ").title() for c in conflicts)
        conflict_html = (
            f'<div style="margin-top:10px;background:#2d1f00;border:1px solid #f0883e;'
            f'border-radius:6px;padding:6px 10px;font-size:.76rem;color:#f0883e;">'
            f'⚠ Conflicts detected in: {names} — verify before advising</div>'
        )

    items = [
        ("PED Waiting Period",  summary.get("ped_waiting_period", "—")),
        ("Co-pay",              summary.get("copay", "—")),
        ("Room Rent",           summary.get("room_rent", "—")),
        ("Portability",         summary.get("portability", "—")),
        ("Renewability",        summary.get("renewability", "—")),
        ("Maternity",           summary.get("maternity", "—")),
    ]
    grid_html = "".join(
        f'<div class="summary-item">'
        f'<div class="summary-item-label">{lbl}</div>'
        f'<div class="summary-item-value">{v}</div>'
        f'</div>'
        for lbl, v in items
    )

    st.markdown(
        f'<div class="summary-card">'
        f'<div class="summary-title">Policy Risk Summary</div>'
        f'<div class="summary-tier" style="color:{tier_color}">Coverage: {tier}</div>'
        f'<div class="summary-conf">Overall Confidence: {conf}% '
        f'({summary.get("fields_high",0)} high · {summary.get("fields_medium",0)} medium · '
        f'{summary.get("fields_not_found",0)} not found / {summary.get("fields_total",0)} total)</div>'
        f'<div class="summary-grid">{grid_html}</div>'
        f'{conflict_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _citation_badges(sources: list[dict]) -> str:
    html = ""
    for s in sources:
        clause = f", {s['clause']}" if s.get("clause") else ""
        label = f"📄 {s['source']}  p.{s['page']} ~l.{s['line']}{clause}"
        page = s.get("page")
        src_name = s.get("source", "")
        if page and src_name:
            html += (
                f'<a class="cite-badge" href="{_pdf_url(src_name, page)}" '
                f'target="_blank" rel="noopener" title="Open PDF at page {page}">{label}</a>'
            )
        else:
            html += f'<span class="cite-badge">{label}</span>'
    return html


def _run_query(question: str) -> None:
    if st.session_state.chain is None:
        st.warning("Please select at least one policy from the sidebar first.")
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving …"):
        result = st.session_state.chain.ask(question)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📋 Policy Interrogator")
    st.caption("Local RAG — powered by Mistral + ChromaDB")
    st.divider()

    # --- Upload ---
    st.markdown('<div class="section-header">Upload PDFs</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop one or more policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = []
        for uf in uploaded_files:
            save_path = DATA_DIR / uf.name
            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(uf.read())
                new_files.append(save_path)
            else:
                new_files.append(save_path)  # already saved

        if st.button("⚡ Index uploaded PDFs", use_container_width=True):
            store = get_store()
            ingestor = PDFIngestor(chunk_size=600, overlap=100)
            total_added = 0
            prog = st.progress(0, text="Indexing …")
            for idx, fp in enumerate(new_files):
                prog.progress((idx) / len(new_files), text=f"Parsing {fp.name} …")
                chunks = ingestor.ingest(fp)
                added = store.add_chunks(chunks)
                total_added += added
            prog.progress(1.0, text="Done!")
            st.success(f"Added {total_added} new chunks across {len(new_files)} file(s).")
            # Invalidate chain
            st.session_state.chain = None
            st.session_state.messages = []

    st.divider()

    # --- Source selector ---
    store = get_store()
    sources = store.list_sources()

    st.markdown('<div class="section-header">Active Policies</div>', unsafe_allow_html=True)

    if sources:
        selected = st.multiselect(
            "Select policies to query / compare:",
            options=sources,
            default=st.session_state.selected_sources or sources[:1],
            label_visibility="collapsed",
        )

        if selected != st.session_state.selected_sources:
            st.session_state.selected_sources = selected
            st.session_state.chain = None
            st.session_state.messages = []

        compare_mode = len(selected) > 1
        st.session_state.compare_mode = compare_mode

        if compare_mode:
            st.info(f"Comparing {len(selected)} policies.")

        # Build chain lazily
        if st.session_state.chain is None and selected and store.count() > 0:
            with st.spinner("Loading Mistral …"):
                st.session_state.chain = get_chain(selected)
    else:
        st.info("Upload and index a PDF to get started.")

    st.divider()

    # --- Stats ---
    st.markdown('<div class="section-header">Store Stats</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Chunks", store.count())
    col2.metric("Policies", len(sources))

    if st.button("🔄 Reset chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chain:
            st.session_state.chain.reset_memory()
        st.rerun()

    # ------------------------------------------------------------------
    # DEBUG SECTION
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("🔍 Debug Retrieval", expanded=False):
        debug_q = st.text_input("Enter a question to debug:", key="debug_question")
        if st.button("Run Debug", use_container_width=True, key="debug_run"):
            if st.session_state.chain is None:
                st.warning("No chain loaded. Please select a policy first.")
            elif not debug_q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running debug... (check terminal for detailed logs)"):
                    # Call debug method and get the output string
                    debug_output = st.session_state.chain.debug_retrieval(debug_q)
                    st.session_state.debug_output = debug_output
                    st.success("Debug complete. See terminal and the code block below.")
        if st.session_state.debug_output:
            st.code(st.session_state.debug_output, language="text")

# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tab_chat, tab_attrs, tab_compare = st.tabs([
    "💬 Chat",
    "📊 Policy Attributes",
    "⚖️ Compare Policies",
])

# ===========================================================================
# TAB 1 — CHAT
# ===========================================================================
with tab_chat:
    selected_sources = st.session_state.get("selected_sources", [])

    if selected_sources:
        st.caption("Querying: " + " · ".join(f"**{s}**" for s in selected_sources))
    else:
        st.caption("No policy selected — use the sidebar.")

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                st.markdown(_citation_badges(msg["sources"]), unsafe_allow_html=True)
            st.markdown("")

    # Quick questions (shown only before first message)
    if not st.session_state.messages and store.count() > 0:
        st.markdown("#### Suggested questions")
        cols = st.columns(2)
        for i, q in enumerate(QUICK_QUESTIONS):
            if cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
                _run_query(q)
                st.rerun()

    # Quick-prompt passthrough (set by comparison tab)
    if st.session_state.get("_pending_prompt"):
        q = st.session_state.pop("_pending_prompt")
        _run_query(q)
        st.rerun()

    # Input
    # Input - Streaming version
    if store.count() == 0:
        st.warning("Index at least one PDF using the sidebar to start asking questions.")
    else:
        user_input = st.chat_input("Ask anything about the selected policy/policies …")
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get chain
            chain = st.session_state.chain

            if chain is None:
                st.warning("Please select at least one policy from the sidebar first.")
                st.session_state.messages.pop()  # Remove user message
            else:
                # Show assistant response with streaming
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    sources = []


                    try:
                        print("=== Testing non-streaming ===")
                        result = chain.ask(user_input)
                        print(result["answer"])
                        print("=============================")
                    except Exception as e:
                        print("Non-streaming failed:", repr(e))

                    try:
                        # Stream the response
                        for chunk in chain.ask_stream(user_input):
                            if chunk["type"] == "text":
                                full_response += chunk["content"]
                                # Update placeholder with streaming text and cursor
                                response_placeholder.markdown(full_response + "▌")
                            elif chunk["type"] == "sources":
                                sources = chunk["sources"]

                        # Final response without cursor
                        response_placeholder.markdown(full_response)

                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources,
                        })

                        # Show citations
                        if sources:
                            st.markdown(_citation_badges(sources), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        # Remove the user message if failed
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()

                    st.rerun()

# ===========================================================================
# TAB 2 — POLICY ATTRIBUTES
# ===========================================================================
with tab_attrs:
    selected_sources = st.session_state.get("selected_sources", [])

    if not selected_sources:
        st.info("Select a policy in the sidebar to extract its key attributes.")
    else:
        for src in selected_sources:
            st.markdown(f"### {src}")

            cached = st.session_state.extracted_attrs.get(src)

            col_btn, col_note = st.columns([1, 3])
            if col_btn.button(f"Extract attributes", key=f"extract_{src}"):
                with st.spinner(f"Analysing {src} …"):
                    tmp_chain = get_chain([src])
                    attrs = tmp_chain.extract_attributes(source_filter=[src])
                    st.session_state.extracted_attrs[src] = attrs
                    cached = attrs

            if cached:
                if "_error" in cached:
                    with st.expander("Raw LLM output (JSON parse failed)"):
                        st.code(cached.get("_raw", ""))
                else:
                    # Summary card at top
                    summary = cached.get("_summary")
                    if summary:
                        _summary_card(summary)

                    sections = {
                        "Policy Overview": [
                            "policy_name", "insurer", "sum_insured_options",
                            "policy_tenure", "lifetime_renewability",
                            "free_look_period_days", "grace_period_days",
                        ],
                        "Waiting Periods": [
                            "waiting_period_initial_days",
                            "waiting_period_ped_months",
                            "waiting_period_specific_illness_months",
                        ],
                        "Co-pay & Sub-limits": [
                            "copay_applicable", "copay_percentage",
                            "copay_conditions", "room_rent_sublimit", "icu_sublimit",
                        ],
                        "Coverage": [
                            "inpatient_covered", "daycare_covered", "domiciliary_covered",
                            "maternity_covered", "ambulance_covered", "organ_donor_covered",
                            "pre_hospitalisation_days", "post_hospitalisation_days",
                        ],
                        "Claims & Renewals": [
                            "cashless_available", "network_hospitals",
                            "claim_settlement_days", "portability_available", "ncb_benefit",
                        ],
                        "Permanent Exclusions": ["permanent_exclusions"],
                    }

                    for section_title, keys in sections.items():
                        st.markdown(
                            f'<div class="section-header">{section_title}</div>',
                            unsafe_allow_html=True,
                        )
                        cols = st.columns(min(len(keys), 3))
                        for i, key in enumerate(keys):
                            with cols[i % len(cols)]:
                                _attr_card(
                                    ATTR_LABELS.get(key, key),
                                    cached.get(key),
                                    source=src,
                                )

                    # Dynamic / policy-specific partner-relevant attributes
                    dynamic = cached.get("_dynamic", {})
                    if dynamic:
                        st.markdown(
                            '<div class="section-header">Policy-Specific Features</div>',
                            unsafe_allow_html=True,
                        )
                        cols = st.columns(min(len(dynamic), 3))
                        for i, (key, val) in enumerate(dynamic.items()):
                            with cols[i % min(len(dynamic), 3)]:
                                _attr_card(
                                    key.replace("_", " ").title(),
                                    val,
                                    source=src,
                                )

            else:
                st.caption("Click **Extract attributes** to auto-analyse this policy.")

            if len(selected_sources) > 1:
                st.divider()

# ===========================================================================
# TAB 3 — COMPARE POLICIES
# ===========================================================================
with tab_compare:
    selected_sources = st.session_state.get("selected_sources", [])

    if len(selected_sources) < 2:
        st.info("Select **2 or more** policies in the sidebar to compare them.")
    else:
        st.markdown(f"### Side-by-side: {' vs '.join(selected_sources)}")

        # Make sure attributes are extracted for all selected
        missing = [s for s in selected_sources if s not in st.session_state.extracted_attrs]
        if missing:
            if st.button("📊 Extract all attributes for comparison", use_container_width=True):
                prog = st.progress(0)
                for idx, src in enumerate(missing):
                    prog.progress(idx / len(missing), text=f"Extracting {src} …")
                    tmp = get_chain([src])
                    st.session_state.extracted_attrs[src] = tmp.extract_attributes(source_filter=[src])
                prog.progress(1.0, text="Done!")
                st.rerun()
        else:
            # Build comparison table
            import pandas as pd

            rows = []
            for key, label in ATTR_LABELS.items():
                row = {"Attribute": label}
                for src in selected_sources:
                    attrs = st.session_state.extracted_attrs.get(src, {})
                    row[src] = _render_value(attrs.get(key))
                rows.append(row)

            df = pd.DataFrame(rows).set_index("Attribute")

            # Highlight differences
            def highlight_diff(row):
                vals = row.values
                if len(set(str(v) for v in vals)) > 1:
                    return ["background-color: #2d1f00; color: #ffa657"] * len(vals)
                return [""] * len(vals)

            styled = df.style.apply(highlight_diff, axis=1)
            st.dataframe(styled, use_container_width=True, height=600)
            st.caption("🟠 Highlighted rows have differing values across policies.")

            st.divider()
            st.markdown("### Ask a comparison question")
            cmp_q = st.text_input(
                "e.g. Which policy has a shorter PED waiting period?",
                key="cmp_input",
            )
            if st.button("Ask", key="cmp_ask") and cmp_q:
                st.session_state["_pending_prompt"] = cmp_q
                # Switch to chat tab by triggering rerun — user sees answer in Chat tab
                st.info("Answer will appear in the **Chat** tab.")
                _run_query(cmp_q)
                st.rerun()
