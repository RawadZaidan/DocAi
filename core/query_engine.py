import os
import streamlit as st
from openai import OpenAI
import tiktoken


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

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


class QueryEngine:
    def estimate_tokens(self, text: str) -> int:
        enc = _get_tokenizer()
        return len(enc.encode(text))

    def build_context(self, docs: dict, max_tokens: int = 180_000) -> tuple:
        sorted_docs = sorted(
            docs.items(),
            key=lambda kv: LABEL_PRIORITY.index(kv[1].get("label", "Unknown"))
            if kv[1].get("label", "Unknown") in LABEL_PRIORITY
            else len(LABEL_PRIORITY),
        )

        included = []
        skipped = []
        parts = []
        accumulated_tokens = 0

        for fname, meta in sorted_docs:
            header = (
                f"=== DOCUMENT: {fname} | TYPE: {meta.get('label', 'Unknown')} ===\n"
            )
            footer = f"\n=== END: {fname} ===\n"
            block = header + meta.get("text", "") + footer
            block_tokens = self.estimate_tokens(block)

            if accumulated_tokens + block_tokens > max_tokens:
                skipped.append(fname)
                if "errors" not in st.session_state:
                    st.session_state["errors"] = []
                st.session_state["errors"].append({
                    "file": fname,
                    "error": "Token budget exceeded — not included in Q&A context",
                })
                continue

            parts.append(block)
            included.append(fname)
            accumulated_tokens += block_tokens

        return "\n".join(parts), included, skipped

    def answer(self, question: str, docs: dict) -> dict:
        context, included, skipped = self.build_context(docs, max_tokens=180_000)

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
                        "You are a tender analysis assistant. Your job is to answer questions "
                        "about the provided tender documents accurately and concisely.\n\n"
                        "RULES:\n"
                        "1. Answer using ONLY information found in the provided documents.\n"
                        "2. For every factual claim, cite the source document in brackets: "
                        "[filename.pdf, Page N] or [filename.docx, Section: X].\n"
                        "3. If the answer is not in the documents, respond exactly: "
                        '"Not found in the provided tender documents."\n'
                        "4. Do not guess, infer, or use outside knowledge.\n"
                        "5. Structure your answer clearly. Use bullet points for lists.\n"
                        "6. If multiple documents give conflicting information, flag the conflict explicitly."
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

        return {
            "answer": answer_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "docs_included": included,
            "docs_skipped": skipped,
        }
