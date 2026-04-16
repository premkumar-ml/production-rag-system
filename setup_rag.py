import os

base = os.getcwd()
os.makedirs("src", exist_ok=True)
os.makedirs("config", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)
os.makedirs("data/sample_docs", exist_ok=True)

# src/__init__.py
with open("src/__init__.py", "w") as f:
    f.write("# RAG Portfolio\n")

# requirements.txt
with open("requirements.txt", "w") as f:
    f.write("""langchain==0.2.16
langchain-community==0.2.17
langchain-openai==0.1.25
langgraph==0.2.28
chromadb==0.5.15
sentence-transformers==3.1.1
openai==1.51.0
rank-bm25==0.2.2
pypdf==4.3.1
beautifulsoup4==4.12.3
requests==2.32.3
python-docx==1.1.2
markdown==3.7
ragas==0.1.21
datasets==3.0.1
pyyaml==6.0.2
python-dotenv==1.0.1
tqdm==4.66.5
tiktoken==0.7.0
numpy==1.26.4
pandas==2.2.3
streamlit==1.38.0
""")

# config/prompts.yaml
with open("config/prompts.yaml", "w") as f:
    f.write("""version: "1.2.0"
prompts:
  rag_answer:
    version: "1.2.0"
    template: |
      You are a precise, citation-driven research assistant.
      RULES:
      1. Answer ONLY using information present in the CONTEXT below.
      2. Every factual claim must be followed by its source in brackets: [chunk_id].
      3. If the context does not contain enough information, respond with:
         "I cannot answer this question based on the provided documents."
      4. Never speculate or hallucinate.

      CONTEXT:
      {context}

      QUESTION:
      {question}

      ANSWER (with citations):

  citation_check:
    version: "1.0.1"
    template: |
      You are a strict factual auditor.
      RETRIEVED CHUNKS:
      {context}
      GENERATED ANSWER:
      {answer}
      Return ONLY valid JSON:
      {{"grounded": true, "unsupported_claims": [], "verdict": "PASS", "reason": "explanation"}}

  query_expansion:
    version: "1.0.0"
    template: |
      Generate 3 alternative phrasings of this question for document retrieval.
      Return ONLY a JSON array of strings.
      ORIGINAL QUESTION: {question}
      VARIANTS (JSON array):
""")

# evaluation/golden_dataset.json
with open("evaluation/golden_dataset.json", "w") as f:
    f.write("""[
  {
    "id": "q001",
    "question": "What is Retrieval Augmented Generation (RAG) and why is it used?",
    "ground_truth": "RAG combines information retrieval with language model generation to produce answers grounded in actual document evidence rather than model memorization.",
    "expected_citations": true,
    "category": "conceptual"
  },
  {
    "id": "q002",
    "question": "What chunk size and overlap are recommended for RAG document ingestion?",
    "ground_truth": "Chunks of 500 to 800 tokens with approximately 100 tokens of overlap between consecutive chunks.",
    "expected_citations": true,
    "category": "technical"
  },
  {
    "id": "q003",
    "question": "Why is hybrid retrieval better than vector search alone?",
    "ground_truth": "Vector search understands semantic meaning but BM25 keyword search handles exact term matching. Combining both via RRF gives higher recall and precision.",
    "expected_citations": true,
    "category": "technical"
  },
  {
    "id": "q004",
    "question": "What is the capital of Mars?",
    "ground_truth": "UNANSWERABLE - this information is not in the documents.",
    "expected_citations": false,
    "category": "adversarial"
  }
]
""")

# data/sample_docs/sample_rag_guide.md
with open("data/sample_docs/sample_rag_guide.md", "w") as f:
    f.write("""# Production RAG Systems Guide

## What is RAG?
Retrieval Augmented Generation (RAG) combines information retrieval with large language model generation.
The key differentiator is citations - every answer must trace back to specific retrieved chunks.

## Phase 1: Document Ingestion
Documents should be chunked into 500 to 800 tokens with 100 tokens of overlap.
Store chunks as embeddings in ChromaDB or Weaviate.

## Phase 2: Hybrid Retrieval
Vector search understands semantic meaning. BM25 handles exact term matching.
Combine both using Reciprocal Rank Fusion (RRF).
Use cross-encoder reranking with ms-marco-MiniLM-L-6-v2 model.
The system should decline to answer if context does not support the response.
Store prompts in a version-controlled YAML config file.

## Phase 3: Evaluation
Curate 50 to 200 golden question-answer pairs.
Use RAGAS to measure faithfulness, answer relevancy, context precision, and context recall.
Wire evaluation into CI/CD pipeline - fail the build if quality drops.
""")

# src/ingestion.py
with open("src/ingestion.py", "w") as f:
    f.write('''"""
ingestion.py - Document Ingestion and Chunking
"""
from __future__ import annotations
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import requests
import tiktoken
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    source_type: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            **self.metadata,
        }

class TextChunker:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text, source, source_type, page_number=None, extra_metadata=None):
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.enc.decode(chunk_tokens).strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"{source}:{idx}:{chunk_text[:64]}".encode()
                ).hexdigest()[:16]
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id, text=chunk_text, source=source,
                    source_type=source_type, page_number=page_number,
                    chunk_index=idx, token_count=len(chunk_tokens),
                    metadata=extra_metadata or {},
                ))
                idx += 1
            if end == len(tokens):
                break
            start = end - self.chunk_overlap
        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

class DocumentIngestionPipeline:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest(self, source):
        if source.endswith(".md") or source.endswith(".txt"):
            with open(source, "r", encoding="utf-8") as f:
                text = f.read()
            return self.chunker.chunk(text, source=source, source_type="markdown")
        elif source.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(source)
            chunks = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.extend(self.chunker.chunk(text, source=source, source_type="pdf", page_number=i))
            return chunks
        else:
            logger.warning(f"Unsupported file type: {source}")
            return []

    def ingest_directory(self, directory):
        all_chunks = []
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in {".pdf", ".md", ".txt"}:
                try:
                    all_chunks.extend(self.ingest(str(path)))
                except Exception as e:
                    logger.error(f"Failed on {path}: {e}")
        return all_chunks
''')

# src/retrieval.py
with open("src/retrieval.py", "w") as f:
    f.write('''"""
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
            parts.append(f"[{c['chunk_id']}] SOURCE: {source_label}\\n{c['text']}\\n")
        return "\\n---\\n".join(parts)
''')

# src/citation.py
with open("src/citation.py", "w") as f:
    f.write('''"""
citation.py - Citation Enforcement
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

DECLINE_RESPONSE = (
    "I cannot answer this question based on the provided documents. "
    "The retrieved context does not contain sufficient information."
)

@dataclass
class CitationVerdict:
    grounded: bool
    verdict: str
    unsupported_claims: List[str]
    reason: str
    cited_chunk_ids: List[str]

def extract_cited_chunk_ids(answer_text):
    return list(set(re.findall(r"\\[([a-zA-Z0-9_\\-]{4,20})\\]", answer_text)))

def check_citations_present(answer_text, retrieved_chunk_ids):
    cited = extract_cited_chunk_ids(answer_text)
    return any(c in set(retrieved_chunk_ids) for c in cited)

class CitationEnforcer:
    def __init__(self, llm_client, prompt_config, mode="strict"):
        self.llm = llm_client
        self.prompt_template = prompt_config["template"]
        self.mode = mode

    def verify(self, answer, context, retrieved_chunk_ids):
        if not check_citations_present(answer, retrieved_chunk_ids):
            return CitationVerdict(
                grounded=False, verdict="FAIL",
                unsupported_claims=["No citation references found."],
                reason="Answer did not cite any retrieved chunks.",
                cited_chunk_ids=[])

        prompt = self.prompt_template.format(context=context, answer=answer)
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=512)
            raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip())
            data = json.loads(raw)
            return CitationVerdict(
                grounded=data.get("grounded", False),
                verdict=data.get("verdict", "FAIL"),
                unsupported_claims=data.get("unsupported_claims", []),
                reason=data.get("reason", ""),
                cited_chunk_ids=extract_cited_chunk_ids(answer))
        except Exception as e:
            logger.error(f"Citation check failed: {e}")
            return CitationVerdict(
                grounded=False, verdict="SKIP",
                unsupported_claims=[], reason=str(e),
                cited_chunk_ids=extract_cited_chunk_ids(answer))

    def enforce(self, answer, context, retrieved_chunk_ids):
        verdict = self.verify(answer, context, retrieved_chunk_ids)
        if self.mode == "strict" and verdict.verdict == "FAIL":
            return DECLINE_RESPONSE, verdict
        return answer, verdict
''')

# src/rag_pipeline.py
with open("src/rag_pipeline.py", "w") as f:
    f.write('''"""
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
''')

# app.py
with open("app.py", "w") as f:
    f.write('''"""
app.py - Streamlit RAG Demo UI
Run: streamlit run app.py
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Production RAG System", page_icon="*", layout="wide")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(persist_dir="./chroma_store")

with st.sidebar:
    st.markdown("## RAG System")
    st.divider()
    st.markdown("### Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs or Markdown files",
                                type=["pdf", "md", "txt"], accept_multiple_files=True)
    web_url = st.text_input("Or enter a URL to scrape")
    if st.button("Ingest", use_container_width=True):
        pipeline = load_pipeline()
        sources = []
        if uploaded:
            os.makedirs("./tmp_uploads", exist_ok=True)
            for f in uploaded:
                out_path = f"./tmp_uploads/{f.name}"
                with open(out_path, "wb") as out:
                    out.write(f.getbuffer())
                sources.append(out_path)
        if web_url.strip():
            sources.append(web_url.strip())
        if sources:
            with st.spinner("Ingesting..."):
                n = pipeline.ingest(sources)
            st.success(f"Done! {n} chunks indexed")
        else:
            st.warning("No sources provided.")
    st.divider()
    st.markdown("Built by Premkumar Narla")

st.markdown("# Production RAG System")
st.markdown("Hybrid retrieval - Cross-encoder reranking - Citation enforcement")
st.divider()

query = st.text_input("Ask a question about your documents",
                      placeholder="e.g. What are the recommended chunk sizes for RAG?")

if st.button("Ask") and query.strip():
    pipeline = load_pipeline()
    with st.spinner("Retrieving and generating..."):
        t0 = time.time()
        result = pipeline.query(query.strip())
        elapsed = round(time.time() - t0, 2)
    st.markdown("### Answer")
    st.markdown(f"Grounded: {result['grounded']} | {elapsed}s | {result['num_chunks_retrieved']} chunks")
    st.info(result["answer"])
    if result["sources"]:
        st.markdown("### Sources")
        for src in result["sources"]:
            st.markdown(f"**[{src['chunk_id']}]** {Path(src['source']).name} | score: {src['score']}")
            st.caption(src["preview"])
''')

# .env
with open(".env", "w") as f:
    f.write("OPENAI_API_KEY=your-key-here\n")

print("")
print("=" * 50)
print("  ALL FILES CREATED SUCCESSFULLY!")
print("=" * 50)
print("")
print("Files created:")
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "chroma_store"]]
    level = root.replace(".", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        print(f"{indent}  {file}")
print("")
print("Next step - update your API key:")
print('  notepad .env')
print("")
print("Then run the app:")
print("  streamlit run app.py")
