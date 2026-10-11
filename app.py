"""
DocChat — Chat with your documents (RAG · LangChain · OpenAI · Streamlit)

Run:
    pip install -r requirements.txt
    streamlit run app.py

Key: put OPENAI_API_KEY=sk-... in a .env file (or paste it in the sidebar).
Works with any OpenAI-compatible endpoint: set a Base URL in Settings
(or OPENAI_BASE_URL in .env) and use model names that endpoint supports.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ═══════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
load_dotenv()

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 2rem; max-width: 1000px;}
      .subtitle {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1.2rem;}
      .step {padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 12px; height: 100%;}
      .step b {display: block; margin-bottom: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."
Be clear and concise.

Context:
{context}

Question: {question}

Answer:"""
)

# ═══════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════
defaults = {"vector_store": None, "messages": [], "stats": None}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ═══════════════════════════════════════════════════════════════
# RAG helpers
# ═══════════════════════════════════════════════════════════════
def get_embeddings(api_key, base_url, embed_model):
    """OpenAI embeddings (works with any OpenAI-compatible endpoint)."""
    return OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url)


def read_file(uploaded_file):
    """Turn one uploaded file into a list of Documents (one per PDF page)."""
    name = uploaded_file.name
    if name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return [
            Document(page_content=page.extract_text() or "",
                     metadata={"source": name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": name, "page": 1})]


def build_index(files, chunk_size, chunk_overlap, embeddings):
    """Read -> split -> embed -> store in FAISS."""
    docs = []
    for f in files:
        docs.extend(read_file(f))
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No readable text found. Scanned PDFs (images) are not supported.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    stats = {"files": len(files), "pages": len(docs), "chunks": len(chunks),
             "names": [f.name for f in files]}
    return store, stats


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['source']} · p.{d.metadata['page']}]\n{d.page_content}" for d in docs
    )


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📁 Documents")
    files = st.file_uploader(
        "Upload PDF, TXT or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Settings"):
        model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
        embed_model = st.selectbox("Embedding model", EMBED_MODELS)
        top_k = st.slider("Chunks to retrieve", 1, 8, 4)
        chunk_size = st.slider("Chunk size", 300, 2000, 800, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
        api_key_input = st.text_input(
            "OpenAI API key", type="password",
            placeholder="Uses .env if empty",
        )
        base_url_input = st.text_input(
            "Base URL (optional)",
            placeholder="Only for OpenAI-compatible providers",
        )

    api_key = api_key_input or os.getenv("OPENAI_API_KEY")
    base_url = base_url_input or os.getenv("OPENAI_BASE_URL") or None

    if st.button("Process documents", type="primary",
                 use_container_width=True, disabled=not files):
        try:
            if not api_key:
                raise ValueError("Add your OpenAI API key first (needed to create embeddings).")
            with st.spinner("Reading and indexing..."):
                embeddings = get_embeddings(api_key, base_url, embed_model)
                store, stats = build_index(files, chunk_size, chunk_overlap, embeddings)
                stats["embed_model"] = embed_model
            st.session_state.vector_store = store
            st.session_state.stats = stats
            st.session_state.messages = []
            st.success("Documents ready.")
        except Exception as e:
            st.error(f"Could not process files: {e}")

    if st.session_state.stats:
        st.caption("Indexed: " + ", ".join(st.session_state.stats["names"]))
        st.caption(f"Embeddings: {st.session_state.stats.get('embed_model', '')}")

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG · LangChain · OpenAI · FAISS")

# ═══════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════
st.title("📄 DocChat")
st.markdown('<p class="subtitle">Ask questions and get answers grounded in your own documents.</p>',
            unsafe_allow_html=True)

if not api_key:
    st.warning("Add OPENAI_API_KEY to your .env file, or paste it in **Settings** in the sidebar.")

# ── Empty state ─────────────────────────────────────────────────
if st.session_state.vector_store is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><b>1 · Upload</b>Add PDF, TXT or MD files in the sidebar.</div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="step"><b>2 · Process</b>Click <i>Process documents</i> to index them.</div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="step"><b>3 · Ask</b>Chat below. Answers cite the source pages.</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Stats row ───────────────────────────────────────────────────
s = st.session_state.stats
m1, m2, m3 = st.columns(3)
m1.metric("Files", s["files"])
m2.metric("Pages", s["pages"])
m3.metric("Chunks", s["chunks"])
st.divider()


def show_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['source']}** · page {src['page']}")
            st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))


# ── Chat history ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# ── New question ────────────────────────────────────────────────
if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("OpenAI API key missing.")
            st.stop()
        try:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(question)

            llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)
            chain = PROMPT | llm | StrOutputParser()
            answer = st.write_stream(
                chain.stream({"context": format_docs(docs), "question": question})
            )

            sources = [{"source": d.metadata["source"], "page": d.metadata["page"],
                        "text": d.page_content} for d in docs]
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            text = str(e)
            if "temperature" in text.lower():
                st.error("This model does not accept temperature=0. Use a model like gpt-4o-mini, "
                         "or remove `temperature=0` in the code.")
            elif "401" in text or "invalid_api_key" in text.lower():
                st.error("Your API key was rejected. Check OPENAI_API_KEY.")
            elif "429" in text or "insufficient_quota" in text:
                st.error("Rate limit or quota reached. Check your OpenAI billing/usage.")
            else:
                st.error(f"Something went wrong: {text}")
