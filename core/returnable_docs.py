import json
import os
import re

import pandas as pd
from rapidfuzz import fuzz

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from core.utils import estimate_tokens as _estimate_tokens


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.5-flash-02-23"

LABEL_PRIORITY = [
    "Instructions to Bidders",
    "Legal / Eligibility Forms",
    "Annexes / Returnables",
    "Technical Specifications",
    "Financial Template",
    "Unknown",
]

# Regex fallback patterns grouped by category
_FALLBACK_PATTERNS: dict[str, list[str]] = {
    "Certifications": [
        r'\bISO\s*\d{4,5}(?::\d{4})?\b(?:[\w\s,]{0,50}(?:certificate|certification|certified))?',
        r'\bFDA\b[\w\s,]{0,60}(?:approval|clearance|registration|certificate)',
        r'\bCE\s+mark(?:ing)?\b',
        r'\bcertif(?:icate|ication|ied)\s+[\w\s]{3,50}',
        r'\baccreditation\s+[\w\s]{3,40}',
    ],
    "Forms": [
        r'\bform\s+[\w\-]+',
        r'\bannex\s+[\w\-]+',
        r'\bappendix\s+[\w\-]+',
        r'\bschedule\s+[\w\-]+',
        r'\btemplate\s+[\w\-]+',
    ],
    "Financial Documents": [
        r'\b(?:financial|price|cost|bid|offer)\s+(?:proposal|offer|schedule|template|statement|form)\b',
        r'\bbill\s+of\s+quantities\b',
        r'\bBoQ\b',
        r'\baudited?\s+(?:financial\s+)?statements?\b',
        r'\bbank\s+(?:guarantee|reference|statement|letter)\b',
        r'\bbid\s+(?:security|bond|guarantee)\b',
    ],
    "Legal / Corporate": [
        r'\bregistration\s+(?:certificate|document|extract)\b',
        r'\bpower\s+of\s+attorney\b',
        r'\bauthorization\s+letter\b',
        r'\b(?:articles|certificate)\s+of\s+incorporation\b',
        r'\bdeclaration\s+of\s+(?:non[-\s]debarment|eligibility|conflict|interest|integrity)\b',
        r'\bcompany\s+profile\b',
        r'\btax\s+(?:registration|clearance|compliance)\s*(?:certificate|document)?\b',
    ],
    "Technical Documents": [
        r'\btechnical\s+(?:proposal|offer|specification|compliance|approach|methodology)\b',
        r'\bscope\s+of\s+work\b',
        r'\bwork\s+plan\b',
        r'\b(?:CV|curriculum\s+vitae)\s+of\b[\w\s,]{3,50}',
        r'\bkey\s+(?:personnel|staff|experts?)\b',
        r'\borganizational\s+(?:chart|structure)\b',
    ],
    "Experience / References": [
        r'\b(?:list|statement)\s+of\s+(?:previous|similar|comparable|relevant)\s+(?:projects?|contracts?|works?|experience)\b',
        r'\bpast\s+(?:performance|experience)\b',
        r'\bclient\s+references?\b',
        r'\bcompletion\s+certificates?\b',
        r'\bcontract\s+(?:completion|performance)\s+certificates?\b',
        r'\bsimilar\s+(?:projects?|works?|contracts?)\s+(?:executed|completed|performed|undertaken)\b',
    ],
}

_compiled_fallback: dict[str, list] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _FALLBACK_PATTERNS.items()
}


def _build_focused_context(docs: dict, max_tokens: int = 80_000) -> tuple[str, list[str], list[str]]:
    sorted_docs = sorted(
        docs.items(),
        key=lambda kv: next(
            (i for i, lbl in enumerate(LABEL_PRIORITY) if kv[1].get("label") == lbl), 99
        ),
    )
    parts: list[str] = []
    included: list[str] = []
    skipped: list[str] = []
    total_tokens = 0

    for fname, meta in sorted_docs:
        # Cap each document at 30 000 chars to avoid one file eating the budget
        body = meta.get("text", "")[:30_000]
        block = (
            f"=== DOCUMENT: {fname} | TYPE: {meta.get('label', 'Unknown')} ===\n"
            f"{body}\n"
            f"=== END: {fname} ===\n\n"
        )
        toks = _estimate_tokens(block)
        if total_tokens + toks > max_tokens:
            skipped.append(fname)
            continue
        parts.append(block)
        included.append(fname)
        total_tokens += toks

    return "".join(parts), included, skipped


def _parse_json_from_response(raw: str) -> list[dict]:
    # Strip markdown fences (handles ```json ... ``` and bare ``` ... ```)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw.strip())

    # Isolate the outermost JSON array if prose surrounds it
    array_match = re.search(r"\[[\s\S]*\]", raw)
    candidate = array_match.group(0) if array_match else raw

    # Strategy 1: direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix trailing commas before ] or } (common LLM mistake)
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract each {...} object individually and stitch them together
    objects = []
    for m in re.finditer(r"\{[\s\S]*?\}", candidate):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if objects:
        return objects

    raise ValueError(f"Could not extract valid JSON from model response. First 300 chars: {raw[:300]}")


class ReturnableDocsExtractor:
    def extract(self, docs: dict) -> dict:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if api_key and OpenAI:
            try:
                return self._extract_via_openrouter(docs, api_key)
            except Exception as e:
                result = self._extract_via_regex(docs)
                result["error"] = f"AI extraction failed ({e}); switched to regex."
                return result
        return self._extract_via_regex(docs)

    # ------------------------------------------------------------------
    def _extract_via_openrouter(self, docs: dict, api_key: str) -> dict:
        context, included, skipped = _build_focused_context(docs, max_tokens=80_000)
        all_filenames = list(docs.keys())

        system_prompt = (
            "You are a tender document analyst specialising in procurement compliance. "
            "Your task is to read the supplied tender documents and return a comprehensive, "
            "structured list of every document a bidder must (or should) submit as part of "
            "their bid package. Include forms, annexes, certifications, declarations, "
            "financial templates, experience statements, CVs, and any other returnables "
            "explicitly or implicitly required."
        )

        user_prompt = f"""Analyze the tender documents below and extract ALL required returnable documents.

Return ONLY a valid JSON array of objects. No markdown, no code fences, no prose before or after.
Every object must have exactly these keys (all strings/booleans, no extra keys):
  "doc_name"    – concise title, e.g. "ISO 9001 Certificate" or "Form 1 – Bid Submission Form"
  "category"    – exactly one of: Certifications | Forms | Financial Documents | Legal / Corporate | Technical Documents | Experience / References | Other
  "mandatory"   – true or false (boolean, not string)
  "description" – one sentence: what the document is and why it is required
  "source_file" – filename where this requirement appears

Start your response with [ and end with ]. Do not include any other text.

TENDER DOCUMENTS:
{context}

["""

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content or ""
        items: list[dict] = _parse_json_from_response(raw)

        # Cross-check against actual package filenames
        for item in items:
            item["found_in_package"] = any(
                fuzz.partial_ratio(item.get("doc_name", "").lower(), fn.lower()) >= 70
                for fn in all_filenames
            )

        return {
            "items": items,
            "method": "claude",
            "docs_included": included,
            "docs_skipped": skipped,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "error": None,
        }

    # ------------------------------------------------------------------
    def _extract_via_regex(self, docs: dict) -> dict:
        all_filenames = list(docs.keys())
        items: list[dict] = []

        for fname, meta in docs.items():
            text = meta.get("text", "")
            for category, patterns in _compiled_fallback.items():
                for pattern in patterns:
                    for m in pattern.finditer(text):
                        line_start = text.rfind("\n", 0, m.start()) + 1  # 0 when no prior \n
                        line_end = text.find("\n", m.end())
                        if line_end == -1:
                            line_end = len(text)
                        line = text[line_start:line_end].strip()
                        # Truncate at first sentence boundary after the match
                        dot = line.find(".", m.end() - line_start)
                        raw = line[:dot + 1].strip() if dot != -1 else line
                        if len(raw) < 5:
                            continue
                        # Truncate at word boundary, max 150 chars
                        cleaned = re.sub(r"\s+", " ", raw).strip()
                        if len(cleaned) > 150:
                            cut = cleaned[:150].rsplit(" ", 1)[0]
                            cleaned = cut + "…"
                        doc_name = cleaned.title()[:160]

                        is_dupe = any(
                            fuzz.ratio(doc_name.lower(), ex["doc_name"].lower()) >= 80
                            for ex in items
                        )
                        if is_dupe:
                            continue

                        found = any(
                            fuzz.partial_ratio(doc_name.lower(), fn.lower()) >= 70
                            for fn in all_filenames
                        )
                        items.append({
                            "doc_name": doc_name,
                            "category": category,
                            "mandatory": True,
                            "description": f"Identified via pattern match in {fname}.",
                            "source_file": fname,
                            "found_in_package": found,
                        })

        items.sort(key=lambda x: (x["found_in_package"], x["category"], x["doc_name"]))
        return {
            "items": items,
            "method": "regex",
            "docs_included": all_filenames,
            "docs_skipped": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "error": None,
        }

    # ------------------------------------------------------------------
    def to_dataframe(self, result: dict) -> pd.DataFrame:
        items = result.get("items", [])
        if not items:
            return pd.DataFrame(
                columns=["Document", "Category", "Mandatory", "Description", "Source File", "Status"]
            )
        rows = [
            {
                "Document": item.get("doc_name", ""),
                "Category": item.get("category", "Other"),
                "Mandatory": item.get("mandatory", True),
                "Description": item.get("description", ""),
                "Source File": item.get("source_file", ""),
                "Status": "✅ Found" if item.get("found_in_package") else "❌ Missing",
            }
            for item in items
        ]
        return pd.DataFrame(rows)
