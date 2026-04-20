"""
Ingestion pipeline: PDF files → chunks → embeddings → FAISS vector DB
Run once (or whenever you add new documents):  python ingest.py
"""
import os
import pickle

from config import settings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(data_dir: str):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            print(f"  Loading {fname}...")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs


def build_index():
    print(f"[ingest] Loading documents from '{settings.data_dir}/'...")
    docs = load_documents(settings.data_dir)
    if not docs:
        print("[ingest] No PDF files found. Add PDFs to the data/ directory.")
        return

    print(f"[ingest] Loaded {len(docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Created {len(chunks)} chunks.")

    print("[ingest] Creating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(settings.vector_db_path, exist_ok=True)
    vectorstore.save_local(settings.vector_db_path)

    chunks_path = os.path.join(settings.vector_db_path, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[ingest] Index saved to '{settings.vector_db_path}/'.")
    print(f"[ingest] Done. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    build_index()
