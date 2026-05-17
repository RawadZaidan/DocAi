import tiktoken

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def estimate_tokens(text: str) -> int:
    try:
        return len(get_tokenizer().encode(text))
    except Exception:
        return len(text) // 4


def strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if "```" not in raw:
        return raw
    inner = raw.split("```", 2)[1]
    if inner.lower().startswith("json"):
        inner = inner[4:]
    return inner.strip()
