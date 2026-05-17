import re
import textwrap
import pandas as pd
from rapidfuzz import fuzz


KEYWORD_PATTERN = re.compile(
    r'\b(shall|must|required|mandatory|obligatory)\b', re.IGNORECASE
)
HEADING_PATTERN = re.compile(
    r'^([A-Z][A-Z\s]{3,}|.*:\s*$|\d+\.\s+\S.*)'
)


class RequirementMatrix:
    def extract(self, docs: dict) -> list:
        requirements = []
        req_counter = 0

        for fname, meta in docs.items():
            text = meta.get("text", "")
            lines = text.splitlines()

            for line_idx, line in enumerate(lines):
                if not KEYWORD_PATTERN.search(line):
                    continue

                sentences = re.split(r'(?<=[.!?])\s+', line)
                for sentence in sentences:
                    if not KEYWORD_PATTERN.search(sentence):
                        continue

                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue

                    # Truncate at word boundary to 300 chars
                    if len(sentence) > 300:
                        truncated = sentence[:297]
                        last_space = truncated.rfind(" ")
                        sentence = truncated[:last_space] + "..." if last_space > 0 else truncated + "..."

                    # Deduplicate
                    is_dupe = any(
                        fuzz.ratio(sentence, r["description"]) > 85
                        for r in requirements
                    )
                    if is_dupe:
                        continue

                    # Determine mandatory flag
                    kw_match = KEYWORD_PATTERN.search(sentence)
                    kw = kw_match.group(1).lower() if kw_match else ""
                    mandatory = kw in ("shall", "must", "mandatory", "obligatory", "required")

                    # Section hint: look back up to 10 lines
                    section_hint = "—"
                    for back in range(1, min(11, line_idx + 1)):
                        prev = lines[line_idx - back].strip()
                        if not prev:
                            continue
                        if (
                            prev.isupper()
                            or prev.endswith(":")
                            or re.match(r'^\d+[\.\)]\s+\S', prev)
                        ):
                            section_hint = prev[:80]
                            break

                    req_counter += 1
                    requirements.append({
                        "req_id": f"REQ-{req_counter:03d}",
                        "description": sentence,
                        "mandatory": mandatory,
                        "source_file": fname,
                        "section_hint": section_hint,
                    })

        return requirements

    def to_dataframe(self, requirements: list) -> pd.DataFrame:
        rows = [
            {
                "Req ID": r["req_id"],
                "Mandatory": r["mandatory"],
                "Description": r["description"],
                "Section": r["section_hint"],
                "Source File": r["source_file"],
            }
            for r in requirements
        ]
        return pd.DataFrame(rows)
