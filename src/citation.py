"""
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
    return list(set(re.findall(r"\[([a-zA-Z0-9_\-]{4,20})\]", answer_text)))

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
