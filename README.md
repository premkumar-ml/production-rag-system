# 🔍 Production-Grade RAG System

**Built by [Premkumar Narla](https://linkedin.com/in/premkumar2905)**  
*AI/ML Engineer · Python · LangGraph · ChromaDB · sentence-transformers · RAGAS*

---

A production-quality Retrieval Augmented Generation system built in three phases — from a working demo to a fully evaluated, CI-gated, enterprise-ready pipeline. This project demonstrates the engineering discipline that separates a RAG demo from a RAG *system*.

## Architecture

```
Documents (PDF / MD / Web)
        │
        ▼
┌─────────────────────┐
│  Ingestion Pipeline │  Token-aware chunking (600 tok / 100 overlap)
└─────────┬───────────┘
          │
    ┌─────┴────┐
    ▼          ▼
ChromaDB    BM25 Index
(Dense)     (Sparse)
    │          │
    └────┬─────┘
         ▼
  RRF Fusion (Reciprocal Rank Fusion)
         │
         ▼
  Cross-Encoder Reranker
  (ms-marco-MiniLM-L-6-v2)
         │
         ▼
  LangGraph Orchestration
  ┌──────┴────────┐
  │   LLM (GPT)  │  Prompt v1.2.0 (versioned YAML)
  └──────┬────────┘
         │
         ▼
  Citation Enforcer
  (decline if not grounded)
         │
         ▼
  Final Answer + Sources
```

## Project Structure

```
rag_portfolio/
├── src/
│   ├── ingestion.py       # Phase 1: PDF/MD/web ingestion + chunking
│   ├── retrieval.py       # Phase 2: BM25 + vector + RRF + reranking
│   ├── citation.py        # Phase 2: Citation enforcement
│   └── rag_pipeline.py    # LangGraph state machine orchestrator
├── evaluation/
│   ├── golden_dataset.json  # Phase 3: 10 manually verified Q&A pairs
│   └── eval_script.py       # Phase 3: RAGAS evaluation + CI exit codes
├── config/
│   └── prompts.yaml         # Versioned prompt registry
├── data/
│   └── sample_docs/         # Sample documents for testing
├── .github/
│   └── workflows/
│       └── eval.yml         # GitHub Actions CI quality gate
├── app.py                   # Streamlit UI
├── requirements.txt
└── .env.example
```

## Phases

### Phase 1 — Fundamentals
- PDF, Markdown, plain text, and web URL ingestion
- Token-accurate chunking (600 tok window, 100 tok overlap via tiktoken)
- Deterministic chunk IDs (SHA-256) for deduplication
- ChromaDB with `all-MiniLM-L6-v2` embeddings
- Basic retrieval with full source citations

### Phase 2 — Production Quality
- **Hybrid retrieval**: BM25 sparse + ChromaDB dense in parallel
- **Reciprocal Rank Fusion** (RRF, k=60) to merge result lists
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores (query, chunk) pairs jointly
- **Citation enforcement**: LLM audits every answer; declines if not grounded
- **Versioned prompts**: `config/prompts.yaml` — bump version on every change
- **LangGraph orchestration**: explicit state machine with conditional edges

### Phase 3 — Shippable
- Golden dataset of 10 verified Q&A pairs (extend to 50-200 for production)
- **RAGAS metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- CI/CD quality gate via GitHub Actions — fails the build on regression
- PR comments with evaluation scores automatically posted

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/premkumar2905/production-rag
cd rag_portfolio
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Ingest sample documents
python -c "
from src.rag_pipeline import RAGPipeline
p = RAGPipeline()
n = p.ingest_directory('./data/sample_docs')
print(f'Ingested {n} chunks')
"

# 4. Query the system
python -c "
from src.rag_pipeline import RAGPipeline
p = RAGPipeline()
result = p.query('What chunk size is recommended for RAG?')
print(result['answer'])
print('Sources:', [s['chunk_id'] for s in result['sources']])
"

# 5. Run evaluation
python evaluation/eval_script.py --verbose

# 6. Launch Streamlit UI
streamlit run app.py
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Token-based chunking via tiktoken | Deterministic, model-aware boundaries vs. character splits |
| BM25 + vector hybrid | Semantic recall + exact-term precision; neither alone is sufficient |
| RRF over weighted averaging | Parameter-free fusion; robust to score scale differences |
| Cross-encoder reranking | Jointly scores (query, chunk) — dramatically improves precision |
| Citation enforcement | Trustworthy answers > plausible-sounding answers |
| YAML prompt registry | Prompts are architecture; they deserve versioning and changelogs |
| RAGAS in CI | Prevents quality regressions from shipping undetected |

## Evaluation Thresholds

| Metric | Threshold | Description |
|---|---|---|
| Faithfulness | ≥ 0.75 | Claims grounded in retrieved context |
| Answer Relevancy | ≥ 0.70 | Answer addresses the actual question |
| Context Precision | ≥ 0.65 | Retrieved chunks are relevant |
| Context Recall | ≥ 0.60 | Context covers the ground truth |

## Tech Stack

- **Orchestration**: LangGraph 0.2 + LangChain 0.2
- **Vector Store**: ChromaDB 0.5 (persistent, cosine similarity)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **Reranking**: sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Keyword Search**: rank-bm25 (BM25Okapi)
- **LLM**: OpenAI GPT-4o-mini
- **Evaluation**: RAGAS 0.1
- **UI**: Streamlit 1.38

---

*This project demonstrates end-to-end ML engineering discipline: ingestion → retrieval → generation → citation enforcement → automated evaluation → CI quality gates.*
