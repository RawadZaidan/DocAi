import os
import json
import streamlit as st
from openai import OpenAI
from core.utils import estimate_tokens, strip_json_fences, get_tokenizer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.5-flash-02-23"

LEGAL_LABEL_KEYWORDS = {
    "Contract / Agreement": [
        "agreement", "contract", "between the parties", "whereas",
        "in consideration of", "effective date", "hereinafter",
    ],
    "NDA / Confidentiality": [
        "confidential", "non-disclosure", "nda", "proprietary information",
        "trade secret", "confidentiality",
    ],
    "MOU / Letter of Intent": [
        "memorandum of understanding", "letter of intent", "mou", "loi",
        "framework agreement", "cooperation agreement",
    ],
    "Policy / Regulation": [
        "policy", "procedure", "regulation", "guideline",
        "code of conduct", "standard operating",
    ],
    "License Agreement": [
        "license", "licence", "intellectual property", "ip rights",
        "grant of rights", "royalt", "sublicens",
    ],
    "Service Agreement / SOW": [
        "service agreement", "statement of work", "sow",
        "professional services", "scope of services", "deliverable",
    ],
    "Corporate / Legal Forms": [
        "articles of incorporation", "bylaws", "resolution",
        "power of attorney", "shareholder", "board of directors",
    ],
}

class LegalAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )

    def classify_documents(self, docs: dict) -> dict:
        for fname, meta in docs.items():
            text = meta["text"]
            snippet = (text[:3000] + text[-1000:]).lower()
            best_label = "Other Legal"
            best_score = 0.0
            for label, keywords in LEGAL_LABEL_KEYWORDS.items():
                if not keywords:
                    continue
                matched = sum(1 for kw in keywords if kw in snippet)
                score = matched / len(keywords)
                if score > best_score:
                    best_score = score
                    best_label = label
            docs[fname]["label"] = best_label
            docs[fname]["confidence"] = round(best_score, 3)
        return docs

    def build_context(self, docs: dict, max_tokens: int = 150_000) -> tuple:
        enc = get_tokenizer()
        included = []
        skipped = []
        parts = []
        acc = 0
        for fname, meta in docs.items():
            header = f"=== DOCUMENT: {fname} | TYPE: {meta.get('label', 'Unknown')} ===\n"
            footer = f"\n=== END: {fname} ===\n"
            body = meta.get("text", "")
            block = header + body + footer
            tok = estimate_tokens(block)
            if acc + tok <= max_tokens:
                parts.append(block)
                included.append(fname)
                acc += tok
            else:
                available = max_tokens - acc
                if available > 500:
                    keep = max(0, available - estimate_tokens(header + footer) - 50)
                    truncated = enc.decode(enc.encode(body)[:keep])
                    parts.append(header + truncated + "\n[... TRUNCATED ...]\n" + footer)
                    included.append(f"{fname} (partial)")
                    acc = max_tokens
                else:
                    skipped.append(fname)
                    if "errors" not in st.session_state:
                        st.session_state["errors"] = []
                    st.session_state["errors"].append({
                        "file": fname,
                        "error": "Token budget exceeded — not included in Legal analysis context",
                    })
        return "\n".join(parts), included, skipped

    def analyze(self, docs: dict) -> dict:
        context, included, skipped = self.build_context(docs)

        system = (
            "You are a specialized legal document analyst. Analyze the provided documents and extract structured information.\n\n"
            "Return ONLY a raw JSON object (no markdown fences, no text before or after). Use this exact structure:\n"
            "{\n"
            '  "summary": "2-3 sentence overview of all documents",\n'
            '  "parties": [{"name": "...", "role": "...", "description": "..."}],\n'
            '  "key_dates": [{"event": "...", "date": "...", "source": "..."}],\n'
            '  "obligations": [{"party": "...", "obligation": "...", "mandatory": true, "source": "..."}],\n'
            '  "risk_clauses": [{"type": "...", "clause_excerpt": "...", "risk_level": "High", "source": "...", "explanation": "..."}],\n'
            '  "defined_terms": [{"term": "...", "definition": "..."}]\n'
            "}\n\n"
            "Risk types to flag: Indemnification, Limitation of Liability, Penalty/Liquidated Damages, "
            "Termination Rights, Force Majeure, Confidentiality Breach, IP Assignment, Dispute Resolution, "
            "Governing Law, Non-Compete, Payment Terms.\n"
            "Risk levels: High (significant financial or legal exposure), Medium (notable but manageable), Low (standard clause).\n\n"
            "Extract real content. If a section has nothing to extract, return an empty array."
        )

        user = f"LEGAL DOCUMENTS:\n\n{context}\n\n---\n\nAnalyze and return structured JSON."

        _empty = {"parties": [], "key_dates": [], "obligations": [], "risk_clauses": [], "defined_terms": []}
        response = None
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=4000,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = strip_json_fences(response.choices[0].message.content or "")
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            data = {"summary": "AI response could not be parsed as JSON.", "parse_error": str(e), **_empty}
        except Exception as e:
            data = {"summary": "", "error": str(e), **_empty}

        data["docs_included"] = included
        data["docs_skipped"] = skipped
        data["input_tokens"] = response.usage.prompt_tokens if response and response.usage else 0
        data["output_tokens"] = response.usage.completion_tokens if response and response.usage else 0
        return data

    def answer(self, question: str, docs: dict) -> dict:
        context, included, skipped = self.build_context(docs)
        response = None
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal document analysis assistant.\n\n"
                            "RULES:\n"
                            "1. Answer using ONLY information from the provided documents.\n"
                            "2. Cite the source for every factual claim: [filename, Section X].\n"
                            "3. If the answer is absent: 'Not found in the provided documents.'\n"
                            "4. Describe what the documents say — do not provide legal advice.\n"
                            "5. Flag ambiguities or conflicting provisions explicitly."
                        ),
                    },
                    {"role": "user", "content": f"DOCUMENTS:\n\n{context}\n\n---\n\nQUESTION: {question}"},
                ],
            )
            return {
                "answer": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "docs_included": included,
                "docs_skipped": skipped,
            }
        except Exception as e:
            return {
                "answer": f"Error: {e}",
                "input_tokens": 0,
                "output_tokens": 0,
                "docs_included": included,
                "docs_skipped": skipped,
            }
