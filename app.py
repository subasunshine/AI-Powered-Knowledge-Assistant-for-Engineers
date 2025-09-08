import os
import faiss
import numpy as np
import requests
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ------------------------------
# Config
# ------------------------------
DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
VSTORE_DIR = DATA_DIR / "vectorstore"
VSTORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VSTORE_DIR / "faiss.index"
TEXTS_PATH = VSTORE_DIR / "texts.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

# ------------------------------
# Utilities
# ------------------------------
def load_index():
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
        return index, texts
    else:
        dim = 384
        index = faiss.IndexFlatL2(dim)
        return index, []

def save_index(index, texts):
    faiss.write_index(index, str(INDEX_PATH))
    np.save(TEXTS_PATH, np.array(texts, dtype=object))

def embed_texts(texts, model):
    return model.encode(texts, convert_to_numpy=True)

def ingest_pdf(pdf_file, index, texts, model):
    pdf_path = DOCS_DIR / pdf_file.name
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
    embeddings = embed_texts(chunks, model)
    index.add(embeddings)
    texts.extend(chunks)
    save_index(index, texts)
    return len(chunks)

def query_rag(question, index, texts, model, chat_history, top_k=3):
    if len(texts) == 0:
        return "No documents ingested yet.", []

    q_emb = embed_texts([question], model)
    distances, indices = index.search(q_emb, top_k)
    retrieved_chunks = [texts[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    # Add conversation history
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    prompt = f"""
You are an AI knowledge assistant. Use the given context to answer questions.
Keep answers concise, clear, and based on the context.

Conversation so far:
{history_text}

Context:
{context}

User: {question}
Assistant:
"""

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "No response"), retrieved_chunks
    except Exception as e:
        return f"Error contacting Ollama: {e}", retrieved_chunks

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 AI Knowledge Assistant (RAG + Memory + Ollama)")

embedding_model = SentenceTransformer(MODEL_NAME)
index, texts = load_index()

# Session state for chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - Ingest
st.sidebar.header("📥 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    n = ingest_pdf(uploaded_file, index, texts, embedding_model)
    st.sidebar.success(f"Ingested {n} chunks from {uploaded_file.name}")

if st.sidebar.button("Clear Knowledge Base"):
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    if TEXTS_PATH.exists():
        TEXTS_PATH.unlink()
    index, texts = load_index()
    st.sidebar.warning("Knowledge base cleared!")

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.info("Chat history cleared!")

# Main chat UI
st.subheader("💬 Chat with your Documents")

# Display past messages
for turn in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {turn['question']}")
    st.markdown(f"**🤖 Assistant:** {turn['answer']}")

# User input
user_query = st.text_input("Ask a question")
if user_query:
    answer, sources = query_rag(
        user_query, index, texts, embedding_model, st.session_state.chat_history
    )
    # Save turn
    st.session_state.chat_history.append({"question": user_query, "answer": answer})

    # Display response
    st.markdown(f"**🧑 You:** {user_query}")
    st.markdown(f"**🤖 Assistant:** {answer}")

    # Sources
    with st.expander("📖 Sources used"):
        for i, src in enumerate(sources, 1):
            st.write(f"**{i}.** {src[:200]}...")
