import os
import time
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import ConversationalRetrievalChain

# ---------------- CONFIG ----------------
PDF_PATH = "data/file.pdf"
VECTOR_STORE_PATH = "vector_store"
LOG_PATH = "logs/chat_logs.csv"
OLLAMA_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

os.makedirs("logs", exist_ok=True)

# ---------------- LOGGING UTILS ----------------

def init_log_file():
    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame(columns=[
            "timestamp", "session_id", "user_query",
            "response", "response_time_sec",
            "num_sources", "error"
        ])
        df.to_csv(LOG_PATH, index=False)

def log_interaction(session_id, user_query, response, response_time, num_sources, error=None):
    init_log_file()
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user_query": user_query,
        "response": response,
        "response_time_sec": round(response_time, 2),
        "num_sources": num_sources,
        "error": error
    }
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)

# ---------------- RAG PIPELINE ----------------

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

def build_vector_store():
    st.info("📄 Loading and processing PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    st.success("✅ Vector store created successfully!")

@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

@st.cache_resource
def load_qa_chain():
    vector_store = load_vector_store()
    if vector_store is None:
        return None

    llm = Ollama(model=OLLAMA_MODEL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="📄 Local PDF Chatbot (Ollama + Analytics)", layout="wide")
st.title("📄 Local PDF Chatbot (Ollama + RAG + Analytics)")

# Initialize logging
init_log_file()

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown(f"**LLM Model:** `{OLLAMA_MODEL}`")
    st.markdown(f"**Embedding Model:** `{OLLAMA_EMBED_MODEL}`")
    st.markdown(f"**PDF:** `{PDF_PATH}`")

    if st.button("🔄 Build / Rebuild Vector Store"):
        build_vector_store()
        st.cache_resource.clear()

    st.divider()
    st.header("📊 Analytics")

    if os.path.exists(LOG_PATH):
        logs_df = pd.read_csv(LOG_PATH)

        st.metric("Total Queries", len(logs_df))
        st.metric("Unique Sessions", logs_df["session_id"].nunique())
        st.metric("Average Response Time (s)", round(logs_df["response_time_sec"].mean(), 2))

        st.subheader("🔥 Top 5 Questions")
        top_questions = logs_df["user_query"].value_counts().head(5)
        st.bar_chart(top_questions)

        st.subheader("⏱ Response Time Distribution")
        st.line_chart(logs_df["response_time_sec"])

        st.subheader("📥 Export Logs")
        st.download_button(
            label="Download CSV Logs",
            data=logs_df.to_csv(index=False),
            file_name="chat_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No logs yet.")

# Main App
qa_chain = load_qa_chain()

if qa_chain is None:
    st.warning("⚠️ Vector store not found. Please build it from the sidebar.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Ask questions about your PDF")

user_query = st.chat_input("Type your question here...")

if user_query:
    start_time = time.time()
    error = None
    answer = ""
    sources = []

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({
                    "question": user_query,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                sources = result["source_documents"]
                st.markdown(answer)

                if sources:
                    st.markdown("📚 **Sources:**")
                    for doc in sources:
                        page = doc.metadata.get("page", "unknown")
                        st.markdown(f"- Page {page}")

            except Exception as e:
                error = str(e)
                st.error("❌ An error occurred while processing your query.")

    response_time = time.time() - start_time

    # Log interaction
    log_interaction(
        session_id=st.session_state.session_id,
        user_query=user_query,
        response=answer,
        response_time=response_time,
        num_sources=len(sources),
        error=error
    )

    if not error:
        st.session_state.chat_history.append((user_query, answer))
