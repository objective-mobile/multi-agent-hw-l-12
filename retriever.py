"""
Hybrid retrieval: FAISS semantic search + BM25 lexical search,
merged via EnsembleRetriever, then reranked with a cross-encoder.
Reused from hw5.
"""
import os
import pickle
from functools import lru_cache

from config import settings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


class EnsembleRetriever:
    """Simple ensemble that merges results from multiple retrievers by RRF."""

    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def invoke(self, query: str):
        seen, results = set(), []
        for retriever in self.retrievers:
            for doc in retriever.invoke(query):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        return results


@lru_cache(maxsize=1)
def _load_ensemble() -> EnsembleRetriever:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k_retrieval}
    )

    chunks_path = os.path.join(settings.vector_db_path, "chunks.pkl")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.top_k_retrieval

    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def hybrid_search(query: str) -> list[dict]:
    ensemble = _load_ensemble()
    reranker = _load_reranker()

    candidates = ensemble.invoke(query)
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(float(score), 4),
        }
        for score, doc in ranked[: settings.top_k_rerank]
    ]


def is_index_ready() -> bool:
    index_file = os.path.join(settings.vector_db_path, "index.faiss")
    chunks_file = os.path.join(settings.vector_db_path, "chunks.pkl")
    return os.path.exists(index_file) and os.path.exists(chunks_file)
