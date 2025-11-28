import os
import asyncio
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
else:
    model = None

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("Legal Query Handler — RAG (HF Embeddings + Gemini)")
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

class HuggingFaceEmbedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 16):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.endpoint = f"https://api-inference.huggingface.co/embeddings/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _call(self, inputs):
        resp = requests.post(self.endpoint, headers=self.headers, json={"inputs": inputs}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise Exception(data.get("error"))
        return data

    def embed_documents(self, texts):
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = self._call(batch)
            for item in data:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    embeddings.append(item)
                else:
                    raise Exception("Unexpected embedding item shape")
        return embeddings

    def embed_query(self, text):
        data = self._call([text])
        item = data[0]
        if isinstance(item, dict) and "embedding" in item:
            return item["embedding"]
        if isinstance(item, list):
            return item
        raise Exception("Unexpected embedding item shape")

@st.cache_resource
def get_hf_embedder():
    return HuggingFaceEmbedder(api_key=HF_API_KEY, model=HF_EMBED_MODEL, batch_size=16)

hf_embedder = get_hf_embedder()

with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
    chunk_size = st.number_input("Chunk size (chars)", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)
    if st.button("Process & Index PDF") and pdf_file:
        with st.spinner("Processing and indexing PDF"):
            try:
                reader = PdfReader(pdf_file)
                raw_text = "".join([page.extract_text() or "" for page in reader.pages])
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_text(raw_text)
                if not chunks:
                    st.error("No text extracted from PDF")
                else:
                    docs = [Document(page_content=c) for c in chunks]
                    try:
                        st.session_state.vectorstore = FAISS.from_documents(docs, embedding=hf_embedder)
                        st.success(f"Indexed {len(chunks)} chunks using embedding object")
                    except Exception:
                        try:
                            embeddings = hf_embedder.embed_documents(chunks)
                            if not embeddings:
                                st.error("Embeddings API returned empty list")
                            elif len(embeddings) != len(chunks):
                                st.error(f"Embeddings length {len(embeddings)} != chunks {len(chunks)}")
                            else:
                                st.session_state.vectorstore = FAISS.from_documents(docs, embedding=None, embeddings=embeddings)
                                st.success(f"Indexed {len(chunks)} chunks using precomputed embeddings")
                        except Exception as e:
                            st.exception(f"Failed to create FAISS vectorstore: {e}")
            except Exception as e:
                st.exception(f"Error processing PDF: {e}")
    if st.button("Clear index"):
        st.session_state.vectorstore = None
        st.success("Index cleared")

st.subheader("Ask your legal question")
user_input = st.text_input("Enter your question:")

if st.button("Send") and user_input.strip():
    with st.spinner("Generating..."):
        try:
            context_pdf = ""
            if st.session_state.vectorstore:
                qvec = None
                try:
                    qvec = hf_embedder.embed_query(user_input)
                except Exception:
                    qvec = None
                docs = []
                try:
                    if qvec is not None and hasattr(st.session_state.vectorstore, "similarity_search_by_vector"):
                        docs = st.session_state.vectorstore.similarity_search_by_vector(qvec, k=3)
                    else:
                        try:
                            docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
                        except Exception:
                            docs = st.session_state.vectorstore.similarity_search_by_vector(qvec, k=3) if qvec is not None else []
                except Exception as e:
                    st.error(f"Vector search error: {e}")
                    docs = []
                if docs:
                    try:
                        context_pdf = "\n\n".join([d.page_content for d in docs if hasattr(d, "page_content")])
                    except Exception:
                        context_pdf = "\n\n".join([str(d) for d in docs])
                else:
                    context_pdf = ""
            else:
                context_pdf = ""
            prompt = f"You are a legal assistant.\nThis is not legal advice.\n\nDocument context:\n{context_pdf}\n\nQuestion:\n{user_input}"
            if model is None:
                answer = "Gemini API key missing"
            else:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(model.generate_content_async(prompt))
                    answer = getattr(response, "text", response.get("text") if isinstance(response, dict) else str(response))
                except Exception:
                    response = model.generate_content(prompt)
                    answer = getattr(response, "text", response.get("text") if isinstance(response, dict) else str(response))
            st.session_state.history.append((user_input, answer))
        except Exception as e:
            st.session_state.history.append((user_input, f"Error: {e}"))

if st.session_state.history:
    for u, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {u}")
        st.markdown(f"**Assistant:** {a}")
        st.write("---")

if st.button("Clear chat"):
    st.session_state.history = []
    st.success("Chat cleared")
