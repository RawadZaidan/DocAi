import re
import pandas as pd
from rapidfuzz import fuzz


PATTERNS = [
    r'annex\s+[\w\-]+',
    r'appendix\s+[\w\-]+',
    r'form\s+[\w\-]+',
    r'attachment\s+[\w\-]+',
    r'returnable\s+[\w\-]+',
    r'(?:bidder shall submit|must be accompanied by|required to submit|submit the following)[^\.\n]{5,80}',
    r'(?:original|copy of|certified copy of)\s+[\w\s]{5,60}(?:certificate|registration|declaration|letter)',
]

_compiled = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def _clean(raw: str) -> str:
    cleaned = re.sub(r'\s+', ' ', raw).strip()
    return cleaned[:100].title()


class ChecklistGenerator:
    def generate(self, docs: dict) -> list:
        items = []
        all_filenames = list(docs.keys())

        for fname, meta in docs.items():
            text = meta.get("text", "")
            for pattern in _compiled:
                for match in pattern.findall(text):
                    item_str = _clean(match)
                    if len(item_str) < 5:
                        continue

                    # Deduplicate
                    is_dupe = any(
                        fuzz.ratio(item_str.lower(), ex["item_name"].lower()) >= 85
                        for ex in items
                    )
                    if is_dupe:
                        continue

                    # Check if any file in the package matches this item
                    found = any(
                        fuzz.partial_ratio(item_str.lower(), fn.lower()) >= 75
                        for fn in all_filenames
                    )

                    items.append({
                        "item_name": item_str,
                        "source_file": fname,
                        "found_in_package": found,
                    })

        # Sort: missing items first
        items.sort(key=lambda x: (x["found_in_package"], x["item_name"]))
        return items

    def to_dataframe(self, items: list) -> pd.DataFrame:
        return pd.DataFrame(items, columns=["item_name", "source_file", "found_in_package"])
