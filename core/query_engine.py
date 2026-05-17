import math
import os
import re
from collections import Counter
import streamlit as st
from openai import OpenAI
from core.utils import get_tokenizer, estimate_tokens as _estimate_tokens


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.5-flash-02-23"

LABEL_PRIORITY = [
    "Technical Specifications",
    "Instructions to Bidders",
    "Financial Template",
    "Legal / Eligibility Forms",
    "Annexes / Returnables",
    "Unknown",
]

_STOPWORDS: set = {
    # Common English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "and", "or", "but", "if", "then",
    # Question words
    "what", "when", "where", "how", "who", "which", "that", "this",
    # Verb fillers
    "me", "my", "please", "tell", "give", "find", "show", "mention",
    "state", "note", "describe", "explain", "list",
    # Legal/tender boilerplate
    "herein", "thereof", "notwithstanding", "whereby", "hereunder",
    "pursuant", "aforementioned", "said", "same", "such",
    # Domain-generic (unhelpful for scoring)
    "document", "documents", "tender", "contract", "agreement",
    "section", "article", "clause", "part", "page",
}

class QueryEngine:
    def estimate_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    # ------------------------------------------------------------------
    # Smart document ranking
    # ------------------------------------------------------------------

    def _rank_docs(self, question: str, docs: dict) -> list:
        """Rank documents by relevance to the question.

        Scoring uses three signals:
        1. Keyword frequency (Counter lookup, normalized by log doc length)
        2. Position boost — keywords in the first 20% of a doc score higher
        3. Label-priority boost — higher-priority doc types score higher
        """
        q_words = {
            w for w in re.findall(r"\w+", question.lower())
            if w not in _STOPWORDS and len(w) > 2
        }

        scores = []
        for fname, meta in docs.items():
            text = meta.get("text", "")
            words = re.findall(r"\w+", text.lower())
            counts = Counter(words)
            word_count = max(len(words), 1)

            # Signal 1: frequency, normalized by log(word_count) to remove long-doc bias
            raw_freq = sum(counts[w] for w in q_words)
            normalized = raw_freq / math.log(word_count + 1)

            # Signal 2: position — keywords in the first 20% score an extra 50%
            head_words = re.findall(r"\w+", text[: max(1, len(text) // 5)].lower())
            head_counts = Counter(head_words)
            position_boost = sum(head_counts[w] for w in q_words) * 0.5

            # Signal 3: label priority
            label = meta.get("label", "Unknown")
            priority_idx = (
                LABEL_PRIORITY.index(label) if label in LABEL_PRIORITY else len(LABEL_PRIORITY)
            )
            priority_boost = (len(LABEL_PRIORITY) - priority_idx) * 0.5

            scores.append((fname, normalized + position_boost + priority_boost))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [fname for fname, _ in scores]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        docs: dict,
        max_tokens: int = 180_000,
        question: str = "",
        codemap: str = "",
        mindmap: str = "",
    ) -> tuple:
        """Assemble context within the token budget.

        Always prepends codemap + mindmap (compact overview of all docs).
        Ranks documents by relevance when a question is provided so the most
        useful documents are included in full first. When a document is too
        large to fit in full, includes a partial truncation instead of skipping.
        Documents that cannot fit even partially are represented by codemap only.
        """
        enc = get_tokenizer()

        header_parts = [p for p in [codemap, mindmap] if p]
        header_text = "\n\n".join(header_parts)
        header_tokens = self.estimate_tokens(header_text) if header_text else 0
        remaining_budget = max_tokens - header_tokens

        ordered_names = (
            self._rank_docs(question, docs)
            if question
            else sorted(
                docs,
                key=lambda f: (
                    LABEL_PRIORITY.index(docs[f].get("label", "Unknown"))
                    if docs[f].get("label", "Unknown") in LABEL_PRIORITY
                    else len(LABEL_PRIORITY)
                ),
            )
        )

        included = []
        skipped = []
        parts = []
        accumulated = 0

        for fname in ordered_names:
            if accumulated >= remaining_budget:
                skipped.append(fname)
                continue

            meta = docs[fname]
            doc_header = f"=== DOCUMENT: {fname} | TYPE: {meta.get('label', 'Unknown')} ===\n"
            doc_footer = f"\n=== END: {fname} ===\n"
            body = meta.get("text", "")
            block = doc_header + body + doc_footer
            block_tokens = self.estimate_tokens(block)

            if accumulated + block_tokens <= remaining_budget:
                # Full document fits
                parts.append(block)
                included.append(fname)
                accumulated += block_tokens
            else:
                # Partial inclusion — truncate to remaining budget
                available = remaining_budget - accumulated
                if available > 500:
                    tokens = enc.encode(body)
                    keep = max(0, available - self.estimate_tokens(doc_header + doc_footer) - 50)
                    truncated_body = enc.decode(tokens[:keep])
                    block = (
                        doc_header
                        + truncated_body
                        + "\n[... TRUNCATED — full structure in codemap above ...]\n"
                        + doc_footer
                    )
                    parts.append(block)
                    included.append(f"{fname} (partial)")
                    accumulated = remaining_budget  # budget now exhausted
                else:
                    skipped.append(fname)
                    if "errors" not in st.session_state:
                        st.session_state["errors"] = []
                    st.session_state["errors"].append({
                        "file": fname,
                        "error": "Token budget exhausted — represented in codemap only",
                    })

        full_parts = ([header_text] if header_text else []) + parts
        return "\n\n".join(full_parts), included, skipped

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def answer(self, question: str, docs: dict, codemap: str = "", mindmap: str = "") -> dict:
        context, included, skipped = self.build_context(
            docs,
            max_tokens=180_000,
            question=question,
            codemap=codemap,
            mindmap=mindmap,
        )

        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tender analysis assistant. Answer questions about the provided "
                        "tender documents accurately and concisely.\n\n"
                        "RULES:\n"
                        "1. Answer using ONLY information found in the provided documents.\n"
                        "2. Cite sources in brackets: [filename.pdf, Page N] or [filename.docx, Section: X].\n"
                        "3. If the answer is not in the documents, say exactly: "
                        '"Not found in the provided tender documents."\n'
                        "4. Do not guess or use outside knowledge.\n"
                        "5. Use bullet points for lists.\n"
                        "6. Flag conflicts when multiple documents disagree.\n"
                        "7. The CODEMAP and MIND MAP at the top list all documents — if a document's "
                        "full text was truncated or excluded, note that the codemap shows its structure."
                    ),
                },
                {
                    "role": "user",
                    "content": f"TENDER DOCUMENTS:\n\n{context}\n\n---\n\nQUESTION: {question}",
                },
            ],
        )

        answer_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        corpus_coverage = len(included) / len(docs) if docs else 0.0

        return {
            "answer": answer_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "docs_included": included,
            "docs_skipped": skipped,
            "corpus_coverage": corpus_coverage,
        }
