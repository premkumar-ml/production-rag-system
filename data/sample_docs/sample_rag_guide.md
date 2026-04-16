# Production RAG Systems Guide

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
