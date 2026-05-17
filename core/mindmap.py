import re
from rapidfuzz.fuzz import ratio


def _dedup(items: list, threshold: int = 75) -> list:
    """Semantic deduplication — removes near-duplicate strings using fuzzy ratio."""
    seen = []
    for item in items:
        if item and not any(ratio(item, s) >= threshold for s in seen):
            seen.append(item)
    return seen


def _clean(text: str) -> str:
    """Collapse whitespace and strip to a readable phrase."""
    return re.sub(r"\s+", " ", text).strip()


# Each topic maps to a list of (pattern, group_index) tuples.
# group_index=1 means use captured group 1 (the value after the keyword).
# group_index=0 means use the full match.
_TOPICS: dict = {
    "Project Scope": [
        (r"(?:scope of work|deliverable|objective|purpose of (?:this )?(?:contract|tender|rfp|rfq))[:\s]+([^\.\n]{10,90})", 1),
        (r"(?:the contractor shall|the supplier shall|the service provider shall)\s+([^\.\n]{10,90})", 1),
    ],
    "Key Deadlines": [
        (r"(?:deadline|submission date|closing date|due date|bid closing)[:\s]+([^\.\n]{5,80})", 1),
        (r"(?:valid(?:ity)? (?:period|until|through))[:\s]+([^\.\n]{5,60})", 1),
        (r"(?:offers?|bids?|proposals?)\s+(?:shall be|must be|are)\s+submitted\s+(?:by|before|no later than)\s+([^\.\n]{5,60})", 1),
    ],
    "Financial": [
        (r"(?:total budget|contract value|estimated cost|bid security|performance bond)[:\s]+([^\.\n]{5,80})", 1),
        (r"(?:[\$€£][\d,]+(?:\.\d{2})?|[\d,]+\s*(?:USD|EUR|GBP|LBP))\b[^\.\n]{0,40}", 0),
    ],
    "Eligibility & Requirements": [
        (r"(?:bidder|tenderer|applicant|supplier)\s+(?:must|shall|should|is required to)\s+([^\.\n]{10,90})", 1),
        (r"(?:minimum requirement|eligibility criterion|qualification requirement)[:\s]+([^\.\n]{10,90})", 1),
        (r"(?:registr\w+|certif\w+|licen\w+)\s+(?:is|are|shall be)\s+required\s*([^\.\n]{0,60})", 1),
    ],
    "Submission Items": [
        (r"(?:shall submit|must submit|required to submit|submit the following)[:\s]+([^\.\n]{5,90})", 1),
        (r"(?:the following documents?|the bid shall include|accompanied by)[:\s]+([^\.\n]{5,90})", 1),
    ],
    "Contracting Parties": [
        (r"(?:contracting authority|purchaser|employer|client|procuring entity|the buyer)[:\s]+([^\.\n]{5,70})", 1),
        (r"(?:between|entered into by and between)\s+([^\.\n]{5,70})", 1),
    ],
}


class MindMapBuilder:
    """Hierarchical text mind map of the tender package.

    Extracts key topics using capture-group patterns so only the *value* after
    a trigger keyword is stored (not the keyword itself). Applies rapidfuzz
    semantic deduplication and shows cross-document frequency per topic.
    """

    def build(self, docs: dict) -> str:
        # buckets: topic → list of (cleaned_item, source_fname)
        buckets: dict = {topic: [] for topic in _TOPICS}
        doc_count = len(docs)

        for fname, meta in docs.items():
            text = meta.get("text", "")
            search_text = text[:15000]

            for topic, patterns in _TOPICS.items():
                for pattern, grp in patterns:
                    for m in re.finditer(pattern, search_text, re.IGNORECASE):
                        raw = m.group(grp) if grp and m.lastindex and grp <= m.lastindex else m.group(0)
                        item = _clean(raw)
                        if len(item) >= 6:
                            buckets[topic].append((item, fname))

        lines = [
            "=== TENDER MIND MAP ===",
            f"Structured overview extracted from {doc_count} document(s).",
            "",
        ]

        for topic, entries in buckets.items():
            if not entries:
                continue

            # Deduplicate items semantically
            texts = [e[0] for e in entries]
            unique_texts = _dedup(texts, threshold=75)[:5]

            # Count distinct source documents for this topic
            contributing_docs = {fname for item, fname in entries if item in unique_texts}
            freq_label = f"(found in {len(contributing_docs)} of {doc_count} doc{'s' if doc_count != 1 else ''})"

            lines.append(f"◆ {topic}  {freq_label}")
            for item in unique_texts:
                lines.append(f"  • {item[:105]}")
            lines.append("")

        lines.append("=== END MIND MAP ===")
        return "\n".join(lines)
