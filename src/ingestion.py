"""
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
