"""
retrieval.py - Hybrid Retrieval with BM25 + Vector + Reranking
"""
from __future__ import annotations
import logging
from typing import List
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.ingestion import DocumentChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_portfolio_docs"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class VectorStore:
    def __init__(self, persist_dir="./chroma_store", embed_model=DEFAULT_EMBED_MODEL):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks):
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.to_dict() for c in chunks])

    def query(self, query_text, top_k=20):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, max(self.collection.count(), 1)))
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            chunks.append({
                "chunk_id": meta.get("chunk_id", f"vec_{i}"),
                "text": doc, "source": meta.get("source", ""),
                "page_number": meta.get("page_number"),
                "score": 1 - results["distances"][0][i],
                "retrieval_method": "vector"})
        return chunks

    def count(self):
        return self.collection.count()

class BM25Index:
    def __init__(self):
        self._corpus = []
        self._bm25 = None

    def build(self, chunks):
        self._corpus = [c.to_dict() | {"text": c.text} for c in chunks]
        self._bm25 = BM25Okapi([doc["text"].lower().split() for doc in self._corpus])

    def query(self, query_text, top_k=20):
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query_text.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = dict(self._corpus[idx])
                doc["score"] = float(scores[idx])
                doc["retrieval_method"] = "bm25"
                results.append(doc)
        return results

def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    chunks_by_id = {}
    for ranked_list in result_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id[cid] = chunk
    fused = sorted(chunks_by_id.values(), key=lambda c: scores[c["chunk_id"]], reverse=True)
    for c in fused:
        c["rrf_score"] = scores[c["chunk_id"]]
    return fused

class CrossEncoderReranker:
    def __init__(self, model_name=DEFAULT_RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

class HybridRetriever:
    def __init__(self, vector_store, bm25_index, reranker, candidate_k=20, final_k=5):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.final_k = final_k

    def retrieve(self, query):
        vector_results = self.vector_store.query(query, top_k=self.candidate_k)
        bm25_results = self.bm25_index.query(query, top_k=self.candidate_k)
        fused = reciprocal_rank_fusion([vector_results, bm25_results])
        return self.reranker.rerank(query, fused, top_k=self.final_k)

    def format_context(self, chunks):
        parts = []
        for c in chunks:
            source_label = c["source"]
            if c.get("page_number"):
                source_label += f" (page {c['page_number']})"
            parts.append(f"[{c['chunk_id']}] SOURCE: {source_label}\n{c['text']}\n")
        return "\n---\n".join(parts)
