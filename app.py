import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from extractor import extract_full_document, get_extraction_summary
from vectorless_rag import (answer_vectorless,
                             is_suitable_for_vectorless,
                             extract_pages_dict,
                             build_page_index)
from traditional_rag import (index_document,
                              connect_to_index,
                              answer_traditional,
                              setup_pinecone_index)
from domains import get_domain_config, get_domain_names

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="Universal Document Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ── HEADER ───────────────────────────────────────────
st.title("🧠 Universal Document Intelligence")
st.markdown("""
Upload any document from any industry and get instant 
AI-powered answers — handles text, tables, images and math.
""")
st.divider()

# ── INITIALIZE SESSION STATE ─────────────────────────
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "pages_content" not in st.session_state:
    st.session_state.pages_content = None
if "page_index" not in st.session_state:
    st.session_state.page_index = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = None
if "domain" not in st.session_state:
    st.session_state.domain = None

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # domain selector
    domain_name = st.selectbox(
        "Select Domain:",
        get_domain_names(),
        help="Choose the type of document you're analyzing"
    )
    domain_config = get_domain_config(domain_name)
    st.caption(f"📋 {domain_config['description']}")

    st.divider()

    # RAG mode selector
    rag_mode = st.radio(
        "Select RAG Mode:",
        ["🔵 Vectorless RAG", "🟢 Traditional RAG"],
        help="""
        Vectorless: Send document directly to AI
        Traditional: Search with Pinecone vectors
        """
    )

    st.divider()

    # image extraction toggle
    extract_images = st.toggle(
        "🖼️ Extract & analyze images",
        value=False,
        help="Slower but understands charts and diagrams"
    )

    st.divider()

    # file uploader
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload any PDF",
        type="pdf",
        help="Supports text, tables, images, and math"
    )

    if uploaded_file:
        if st.button("🚀 Process Document",
                     type="primary",
                     use_container_width=True):

            # save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # reset session state for new document
            st.session_state.messages = []
            st.session_state.page_index = None
            st.session_state.vectorstore = None

            # extract document
            with st.spinner("📝 Extracting document..."):
                result = extract_full_document(
                    tmp_path,
                    extract_images=extract_images
                )
                st.session_state.extraction_result = result
                st.session_state.pages_content = extract_pages_dict(result)
                st.session_state.current_file = uploaded_file.name
                st.session_state.rag_mode = rag_mode
                st.session_state.domain = domain_name

            # setup RAG based on mode
            if "Traditional" in rag_mode:
                with st.spinner("🔢 Indexing into Pinecone..."):
                    setup_pinecone_index()
                    vectorstore = index_document(
    tmp_path,
    filename=uploaded_file.name  # use original name
)
                    st.session_state.vectorstore = vectorstore

            elif "Vectorless" in rag_mode:
                # check if page index needed
                suitable, msg = is_suitable_for_vectorless(
                    result["full_text"]
                )
                token_estimate = len(result["full_text"]) // 4

                if token_estimate >= 8000:
                    with st.spinner("📑 Building page index..."):
                        page_index = build_page_index(
                            st.session_state.pages_content,
                            pdf_path=tmp_path
                        )
                        st.session_state.page_index = page_index

            # cleanup temp file
            os.unlink(tmp_path)
            st.success("✅ Document ready!")
            st.rerun()

    st.divider()

    # example questions
    if st.session_state.domain:
        domain_config = get_domain_config(
            st.session_state.domain
        )
        st.header("💡 Example Questions")
        for example in domain_config["examples"]:
            if st.button(example,
                         key=f"ex_{example[:20]}",
                         use_container_width=True):
                st.session_state.pending_question = example

    # clear chat
    if st.button("🗑️ Clear Chat",
                 use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN AREA ────────────────────────────────────────
if st.session_state.extraction_result is None:

    # welcome screen
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **📄 Any Document Type**
        Upload PDFs from any industry
        — medical, legal, financial,
        insurance, or general
        """)

    with col2:
        st.info("""
        **🔍 Two RAG Modes**
        Vectorless RAG for small docs
        Traditional RAG for large docs
        Auto-selects best method
        """)

    with col3:
        st.info("""
        **🖼️ Multimodal**
        Understands text, tables,
        images, charts, diagrams
        and math formulas
        """)

    st.markdown("### 👈 Upload a PDF to get started!")

else:
    result = st.session_state.extraction_result

    # document stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Pages",
                  len(result["pages"]))
    with col2:
        st.metric("📊 Tables",
                  result["table_count"])
    with col3:
        st.metric("🖼️ Images",
                  result["image_count"])
    with col4:
        st.metric("📝 Characters",
                  f"{len(result['full_text']):,}")

    # current settings
    st.caption(
        f"📁 {st.session_state.current_file} | "
        f"🎯 {st.session_state.domain} | "
        f"⚙️ {st.session_state.rag_mode}"
    )

    st.divider()

    # chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                st.caption(
                    f"📍 Sources: "
                    f"{', '.join(message['sources'])}"
                )

            if "tokens" in message and message["tokens"]:
                tokens = message["tokens"]
                total = tokens.get('total', 0)
                method = message.get('method', '')
                st.caption(
                    f"⚡ {total} tokens | "
                    f"🔧 {method}"
                )

    # chat input
    question = st.chat_input(
        "Ask anything about your document..."
    )

    # handle example button clicks
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    # process question
    if question:
        # show user message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # get domain config
        domain_config = get_domain_config(
            st.session_state.domain
        )

        # answer based on RAG mode
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                if "Traditional" in st.session_state.rag_mode:
                    # Traditional RAG
                    if st.session_state.vectorstore:
                        answer, sources, tokens = answer_traditional(
                            question,
                            st.session_state.vectorstore,
                            domain_config["system_prompt"]
                        )
                        method = "Traditional RAG"
                    else:
                        answer = "Please process a document first."
                        sources = []
                        tokens = {}
                        method = ""

                else:
                    # Vectorless RAG
                    answer, tokens, method = answer_vectorless(
                        question,
                        result["full_text"],
                        domain_config["system_prompt"],
                        pages_content=st.session_state.pages_content,
                        page_index=st.session_state.page_index
                    )
                    sources = []

            st.markdown(answer)

            if sources:
                st.caption(
                    f"📍 Sources: {', '.join(sources)}"
                )

            if tokens:
                total = tokens.get('total', 0)
                pages = tokens.get('pages_used', [])
                caption = f"⚡ {total} tokens | 🔧 {method}"
                if pages:
                    caption += f" | 📄 Pages: {pages}"
                st.caption(caption)

        # save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources if sources else [],
            "tokens": tokens,
            "method": method
        })