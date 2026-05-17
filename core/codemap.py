import re
import tiktoken

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                                                          # ISO: 2024-05-15
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",                                         # 15/05/2024
    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",  # 15 May 2024
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b", # May 15, 2024
]

_AMOUNT_PATTERNS = [
    r"[\$€£]\s*[\d,]+(?:\.\d{2})?",                     # $1,000.00 / €500
    r"\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|LBP)\b",   # 1000 USD / 500.00 LBP
    r"\bLBP\s*[\d,]+",                                   # LBP1000
]


class CodemapBuilder:
    """Compact per-document index — prepended to every Q&A context.

    Each document gets a ~60-token entry covering type, structure, and key
    metadata (dates, amounts, deadlines) so the model knows what exists even
    when a document's full text is excluded from the context window.
    """

    def build(self, docs: dict) -> str:
        lines = [
            "=== TENDER CODEMAP ===",
            f"Package: {len(docs)} document(s). Use this index to locate information before reading full text.",
            "",
        ]
        for fname, meta in docs.items():
            label = meta.get("label", "Unknown")
            pages = len(meta.get("pages", []))
            char_count = meta.get("char_count", 0)
            text = meta.get("text", "")
            est_tokens = self.estimate_tokens(text)

            headings = self._extract_headings(text)
            entities = self._extract_entities(text)

            lines.append(f"[{fname}]")
            lines.append(
                f"  Type: {label}  |  Pages/Sheets: {pages}  "
                f"|  Size: {char_count:,} chars (~{est_tokens:,} tokens)"
            )
            if headings:
                lines.append(f"  Sections: {' > '.join(headings)}")
            for e in entities:
                lines.append(f"  {e}")
            lines.append("")

        lines.append("=== END CODEMAP ===")
        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        return len(_get_tokenizer().encode(text))

    def _extract_headings(self, text: str) -> list:
        lines = text.split("\n")
        headings = []
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or len(s) < 4:
                continue

            # Numbered: "1.", "2.3", "1.2.3"
            if re.match(r"^\d+[\.\d]*\s+\w", s) and len(s) < 80:
                headings.append(s[:55])
            # Keyword-led: "Section 3:", "Article IV", "Clause 7", "Part A"
            elif re.match(r"^(?:section|article|clause|part|annex|appendix)\s+[\w\d]+", s, re.IGNORECASE) and len(s) < 80:
                headings.append(s[:55])
            # ALL CAPS (4–55 chars, skip short acronyms)
            elif s.isupper() and 6 < len(s) < 56:
                headings.append(s[:55])
            # Markdown headings ##
            elif s.startswith("##"):
                headings.append(s.lstrip("#").strip()[:55])
            # Underline-style: next non-empty line is all dashes or equals
            elif i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and re.fullmatch(r"[-=]{3,}", nxt) and len(s) < 80:
                    headings.append(s[:55])

            if len(headings) == 8:
                break
        return headings

    def _extract_entities(self, text: str) -> list:
        # Search front + tail of document to catch late-placed metadata
        snippet = text[:10000] + text[-2000:]
        entities = []

        # Dates — collect across all formats, deduplicate
        dates = []
        for pat in _DATE_PATTERNS:
            dates.extend(re.findall(pat, snippet))
        dates = list(dict.fromkeys(d.strip() for d in dates))
        if dates:
            entities.append("Dates: " + ", ".join(dates[:4]))

        # Amounts — symbol-first and number-first patterns
        amounts = []
        for pat in _AMOUNT_PATTERNS:
            amounts.extend(re.findall(pat, snippet))
        amounts = list(dict.fromkeys(a.strip() for a in amounts))
        if amounts:
            entities.append("Amounts: " + ", ".join(amounts[:4]))

        # Deadlines — capture up to 2 distinct phrases
        deadlines = re.findall(
            r"(?:deadline|closing date|submission date|due date)[:\s]+([^\.\n]{5,70})",
            snippet,
            re.IGNORECASE,
        )
        unique_dl = list(dict.fromkeys(d.strip() for d in deadlines))[:2]
        for dl in unique_dl:
            entities.append(f"Deadline: {dl}")

        return entities[:6]
