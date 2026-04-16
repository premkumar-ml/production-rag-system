"""
rag_pipeline.py - LangGraph RAG Pipeline
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict
import yaml
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from src.citation import CitationEnforcer
from src.ingestion import DocumentChunk, DocumentIngestionPipeline
from src.retrieval import BM25Index, CrossEncoderReranker, HybridRetriever, VectorStore

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    query: str
    retrieved_chunks: List[dict]
    context: str
    raw_answer: str
    final_answer: str
    citation_verdict: Optional[dict]
    grounded: bool
    error: Optional[str]

class RAGPipeline:
    def __init__(self, persist_dir="./chroma_store",
                 prompts_path="./config/prompts.yaml",
                 embed_model="all-MiniLM-L6-v2",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 llm_model="gpt-4o-mini",
                 candidate_k=20, final_k=5, citation_mode="strict"):

        with open(prompts_path, "r") as f:
            prompt_cfg = yaml.safe_load(f)
        self.prompts = prompt_cfg["prompts"]
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.vector_store = VectorStore(persist_dir=persist_dir, embed_model=embed_model)
        self.bm25_index = BM25Index()
        self.reranker = CrossEncoderReranker(model_name=rerank_model)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store, bm25_index=self.bm25_index,
            reranker=self.reranker, candidate_k=candidate_k, final_k=final_k)
        self.enforcer = CitationEnforcer(
            llm_client=self.llm.client,
            prompt_config=self.prompts["citation_check"],
            mode=citation_mode)
        self.graph = self._build_graph()

    def ingest(self, sources):
        pipeline = DocumentIngestionPipeline()
        all_chunks = []
        for source in sources:
            all_chunks.extend(pipeline.ingest(source))
        self.vector_store.add_chunks(all_chunks)
        self.bm25_index.build(all_chunks)
        return len(all_chunks)

    def ingest_directory(self, directory):
        pipeline = DocumentIngestionPipeline()
        chunks = pipeline.ingest_directory(directory)
        self.vector_store.add_chunks(chunks)
        self.bm25_index.build(chunks)
        return len(chunks)

    def _node_retrieve(self, state):
        try:
            chunks = self.retriever.retrieve(state["query"])
            context = self.retriever.format_context(chunks)
            return {**state, "retrieved_chunks": chunks, "context": context, "error": None}
        except Exception as e:
            return {**state, "retrieved_chunks": [], "context": "", "error": str(e)}

    def _node_generate(self, state):
        if state.get("error") or not state["context"]:
            return {**state, "raw_answer": "No context available.", "final_answer": ""}
        prompt = self.prompts["rag_answer"]["template"].format(
            context=state["context"], question=state["query"])
        try:
            response = self.llm.invoke(prompt)
            return {**state, "raw_answer": response.content}
        except Exception as e:
            return {**state, "raw_answer": "", "error": str(e)}

    def _node_enforce_citations(self, state):
        chunk_ids = [c["chunk_id"] for c in state["retrieved_chunks"]]
        final_answer, verdict = self.enforcer.enforce(
            state["raw_answer"], state["context"], chunk_ids)
        return {**state, "final_answer": final_answer,
                "citation_verdict": {"grounded": verdict.grounded, "verdict": verdict.verdict,
                                     "reason": verdict.reason, "cited_chunk_ids": verdict.cited_chunk_ids},
                "grounded": verdict.grounded}

    def _should_enforce(self, state):
        return "skip" if (state.get("error") or not state["retrieved_chunks"]) else "enforce"

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("generate", self._node_generate)
        g.add_node("enforce_citations", self._node_enforce_citations)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_conditional_edges("generate", self._should_enforce,
                                {"enforce": "enforce_citations", "skip": END})
        g.add_edge("enforce_citations", END)
        return g.compile()

    def query(self, question):
        initial_state = {"query": question, "retrieved_chunks": [], "context": "",
                         "raw_answer": "", "final_answer": "", "citation_verdict": None,
                         "grounded": False, "error": None}
        final_state = self.graph.invoke(initial_state)
        sources = [{"chunk_id": c["chunk_id"], "source": c["source"],
                    "page": c.get("page_number"),
                    "score": round(c.get("rerank_score", c.get("rrf_score", 0)), 4),
                    "preview": c["text"][:200] + "..."}
                   for c in final_state["retrieved_chunks"]]
        return {"question": question,
                "answer": final_state["final_answer"] or final_state["raw_answer"],
                "grounded": final_state["grounded"],
                "citation_verdict": final_state.get("citation_verdict"),
                "sources": sources,
                "num_chunks_retrieved": len(final_state["retrieved_chunks"])}
