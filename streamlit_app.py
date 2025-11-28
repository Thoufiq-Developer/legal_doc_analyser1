import os
import asyncio
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import DEFAULT_SETTINGS
from chromadb.config import Settings

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEN_EMBED_MODEL = os.getenv("GEN_EMBED_MODEL", "models/embedding-001")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST", "./data/chroma_db")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gen_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gen_model = None

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Legal Chatbot — RAG", layout="wide")
st.title("Legal Query Handler — Gemini + Chroma")
if "history" not in st.session_state:
    st.session_state.history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "client" not in st.session_state:
    st.session_state.client = None

class GeminiEmbeddings:
    def __init__(self, model_name: str = GEN_EMBED_MODEL, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
    def _call_api(self, inputs):
        resp = genai.embeddings.create(model=self.model_name, input=inputs)
        if isinstance(resp, dict) and "data" in resp:
            items = resp["data"]
            return [it.get("embedding") for it in items]
        data = getattr(resp, "data", [])
        out = []
        for it in data:
            emb = getattr(it, "embedding", None) or (it.get("embedding") if isinstance(it, dict) else None)
            out.append(emb)
        return out
    def embed_documents(self, texts):
        if not texts:
            return []
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embs = self._call_api(batch)
            if not embs or len(embs) != len(batch):
                raise Exception("Embedding API returned unexpected shape")
            all_embs.extend(embs)
        return all_embs
    def embed_query(self, text):
        out = self._call_api([text])
        if not out:
            raise Exception("Embedding API returned empty for query")
        return out[0]

@st.cache_resource
def get_gemini_embedder():
    return GeminiEmbeddings(model_name=GEN_EMBED_MODEL, batch_size=16)

@st.cache_resource
def init_chroma(persist_directory: str = CHROMA_PERSIST, collection_name: str = "legal_docs"):
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)
    return client, collection

gemini_embedder = get_gemini_embedder()
client, collection = init_chroma()
st.session_state.client = client
st.session_state.collection = collection

with st.sidebar:
    st.header("Upload and index PDF")
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
                    ids = [f"{pdf_file.name}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": pdf_file.name, "chunk": i} for i in range(len(chunks))]
                    embeddings = gemini_embedder.embed_documents(chunks)
                    try:
                        try:
                            collection.delete(ids=ids)
                        except Exception:
                            pass
                        collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
                        try:
                            client.persist()
                        except Exception:
                            pass
                        st.success(f"Indexed {len(chunks)} chunks in ChromaDB")
                    except Exception as e:
                        st.exception(f"Failed to add docs to Chroma: {e}")
            except Exception as e:
                st.exception(f"Error processing PDF: {e}")
    if st.button("Clear index"):
        try:
            collection.delete()
        except Exception:
            try:
                client.delete_collection("legal_docs")
            except Exception:
                pass
        client, collection = init_chroma()
        st.session_state.client = client
        st.session_state.collection = collection
        st.success("Index cleared")

st.subheader("Ask your legal question")
user_input = st.text_input("Enter your question:")

if st.button("Send") and user_input.strip():
    with st.spinner("Generating..."):
        try:
            context_pdf = ""
            if collection is not None:
                qvec = None
                try:
                    qvec = gemini_embedder.embed_query(user_input)
                except Exception:
                    qvec = None
                docs = []
                try:
                    if qvec is not None:
                        res = collection.query(query_embeddings=[qvec], n_results=3, include=["documents","metadatas","distances"])
                        docs = res.get("documents", [[]])[0]
                    else:
                        res = collection.query(query_texts=[user_input], n_results=3, include=["documents","metadatas","distances"])
                        docs = res.get("documents", [[]])[0]
                except Exception as e:
                    st.error(f"Vector search error: {e}")
                    docs = []
                if docs:
                    try:
                        context_pdf = "\n\n".join([d for d in docs])
                    except Exception:
                        context_pdf = "\n\n".join([str(d) for d in docs])
                else:
                    context_pdf = ""
            else:
                context_pdf = ""
            prompt = f"You are a legal assistant.\nThis is not legal advice.\n\nDocument context:\n{context_pdf}\n\nQuestion:\n{user_input}"
            if gen_model is None:
                answer = "Gemini API key missing"
            else:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(gen_model.generate_content_async(prompt))
                    answer = getattr(response, "text", response.get("text") if isinstance(response, dict) else str(response))
                except Exception:
                    response = gen_model.generate_content(prompt)
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
