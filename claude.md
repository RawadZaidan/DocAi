# Tender Intelligence System (TIS) — CLAUDE.md

> **Model:** `claude-sonnet-4-6`
> **Context strategy:** Full-document stuffing via 1M token window. No RAG. No ChromaDB. No embeddings.
> **UI:** Streamlit
> **Scope:** MVP — working, not perfect. Stub clearly rather than hallucinate.

---

## OVERVIEW

Build a Streamlit MVP called **Tender Intelligence System (TIS)**. This tool ingests any professional tender package (PDF, DOCX, XLSX, CSV) from a local folder, analyzes it using Claude's 1M token context window, and surfaces structured intelligence: document classification, requirement matrix, financial validation, Q&A, and a submission checklist.

**Context philosophy:** Extract all document text upfront. For each Claude API call, pass only the relevant document subset — not the entire corpus — to avoid context rot. Each module assembles its own focused context window.

---

## TECH STACK

```
streamlit
pypdf
python-docx
openpyxl
pandas
rapidfuzz
anthropic
python-dotenv
tiktoken
```

**Rules:**
- `pypdf` only. Never `PyPDF2` (deprecated).
- `anthropic` Python SDK only. No LangChain. No OpenAI.
- Model: `claude-sonnet-4-6`. Never Haiku for complex analysis.
- No ChromaDB. No FAISS. No sentence-transformers. No vector stores of any kind.
- No hardcoded org names: zero mentions of "UNOPS", "GIZ", "UN", "NGO" in logic.

---

## PROJECT STRUCTURE

```
tis/
├── app.py                      # Streamlit entry point
├── .env.example                # ANTHROPIC_API_KEY=
├── requirements.txt
├── README.md
└── core/
    ├── __init__.py
    ├── ingestor.py             # Module 1: Document loading + classification
    ├── requirement_matrix.py  # Module 2: Shall/Must/Required extraction
    ├── financial_parser.py    # Module 3: BoQ parsing + validation
    ├── query_engine.py        # Module 4: Full-context Q&A via Claude
    └── checklist.py           # Module 5: Submission checklist
```

---

## MODULE 1 — `core/ingestor.py`

### `class DocumentIngestor`

**`load_folder(folder_path: str) -> dict`**

Recursively finds all `.pdf`, `.docx`, `.xlsx`, `.csv` files in the given folder.

Returns:
```python
{
  "filename.pdf": {
    "text": str,           # Full extracted text
    "pages": list[str],    # Per-page text (PDF) or per-sheet (Excel)
    "file_type": str,      # "pdf" | "docx" | "xlsx" | "csv"
    "char_count": int,
    "label": str,          # Filled by classify_documents()
    "confidence": float
  }
}
```

Extraction rules:
- **PDF:** `pypdf.PdfReader`. Extract page-by-page. Join with `\n--- Page N ---\n` separators.
- **DOCX:** `python-docx`. Extract paragraphs + table cell text. Preserve heading hierarchy with `##` prefix.
- **XLSX:** `openpyxl`. Each sheet → `pandas.DataFrame` → `.to_markdown(index=False)`. Join sheets with `\n--- Sheet: SheetName ---\n`.
- **CSV:** `pandas.read_csv()` → `.to_markdown(index=False)`.

Every file read must be wrapped in `try/except`. On failure, log to `st.session_state["errors"]` list as `{"file": filename, "error": str(e)}`. Skip failed files and continue.

---

**`classify_documents(docs: dict) -> dict`**

Heuristic classifier — no API call. Scan each document's text (first 3,000 chars + last 1,000 chars) for keyword presence. Assign one label from:

| Label | Keywords (case-insensitive, any match) |
|---|---|
| `Instructions to Bidders` | `instructions to bidders`, `itb`, `rfq`, `rfp`, `submission deadline`, `how to bid`, `bidding procedures` |
| `Technical Specifications` | `technical specification`, `scope of work`, `deliverable`, `shall`, `must`, `performance requirement`, `technical criteria` |
| `Financial Template` | `bill of quantities`, `boq`, `unit price`, `total price`, `rate`, `financial offer`, `price schedule` |
| `Legal / Eligibility Forms` | `eligibility`, `declaration`, `certificate`, `registration`, `authorized signatory`, `power of attorney` |
| `Annexes / Returnables` | `annex`, `appendix`, `form`, `returnable`, `attachment`, `schedule` |
| `Unknown` | Default if no keywords match |

Confidence = (matched_keywords / total_keywords_in_category). Return updated `docs` dict with `label` and `confidence` fields populated.

---

## MODULE 2 — `core/requirement_matrix.py`

### `class RequirementMatrix`

**`extract(docs: dict) -> list[dict]`**

Scan all documents. Use regex to find sentences containing:
```python
r'\b(shall|must|required|mandatory|obligatory)\b'
```
(case-insensitive, `re.IGNORECASE`)

For each match:
- `req_id`: Auto-incremented string `REQ-001`, `REQ-002`, ...
- `description`: Full sentence, max 300 chars, truncated at word boundary with `...`
- `mandatory`: `True` if keyword is `shall` / `must` / `mandatory` / `obligatory`. `False` if `required` (context-dependent — flag as `True` for safety).
- `source_file`: filename
- `section_hint`: Look backward up to 10 lines from the match. Take the last line that is either ALL CAPS, ends with `:`, or starts with a digit followed by `.` (numbered heading). If none found, `"—"`.

Deduplicate: skip sentences that are >85% similar to an already-captured requirement (use `rapidfuzz.fuzz.ratio`).

**`to_dataframe(requirements: list) -> pd.DataFrame`**

Columns: `Req ID | Mandatory | Description | Section | Source File`

---

## MODULE 3 — `core/financial_parser.py`

### `class FinancialParser`

**`parse(docs: dict) -> list[dict]`**

Process all files with `label == "Financial Template"` or `file_type in ["xlsx", "csv"]`.

Returns list of results, one per file:
```python
{
  "filename": str,
  "dataframe": pd.DataFrame,   # Raw parsed table
  "column_map": dict,          # Canonical name → actual column name
  "flags": list[dict]          # Validation issues
}
```

---

**`detect_columns(df: pd.DataFrame) -> dict`**

Use `rapidfuzz.process.extractOne` to match each actual column header against canonical candidates. Only accept if score ≥ 70.

| Canonical | Candidates |
|---|---|
| `description` | `description`, `item`, `scope`, `work item`, `particulars`, `activity` |
| `unit` | `unit`, `uom`, `unit of measure`, `measure` |
| `quantity` | `qty`, `quantity`, `no.`, `nos`, `number` |
| `unit_price` | `unit price`, `rate`, `price`, `cost per unit`, `unit cost`, `unit rate` |
| `total` | `total`, `amount`, `total price`, `subtotal`, `extended price`, `line total` |

---

**`validate(df: pd.DataFrame, column_map: dict) -> list[dict]`**

Run these checks when the relevant columns are detected:

1. **Math check** — For each row: verify `quantity × unit_price ≈ total` within 1% tolerance.
   - Flag format: `{"row": int, "issue": "Math Error", "detail": f"Row {N}: {qty} × {unit_price} = {expected}, found {actual}"}`

2. **Currency inconsistency** — Scan all string cells and column headers for currency symbols/codes: `$`, `€`, `£`, `USD`, `EUR`, `GBP`, `LBP`. If more than one type detected:
   - Flag: `{"row": -1, "issue": "Currency Inconsistency", "detail": f"Multiple currencies detected: {found_currencies}"}`

3. **Empty totals** — Flag rows where `total` is NaN/blank but `quantity` and `unit_price` are present and non-zero:
   - Flag: `{"row": int, "issue": "Missing Total", "detail": "Quantity and unit price present but total is empty"}`

---

## MODULE 4 — `core/query_engine.py`

### `class QueryEngine`

**Architecture:** No vector store. No embeddings. Context-stuffing approach using Claude `claude-sonnet-4-6` with 1M token window.

---

**`estimate_tokens(text: str) -> int`**

Use `tiktoken` with `cl100k_base` encoding as a proxy for Claude token counting. Returns approximate token count.

---

**`build_context(docs: dict, max_tokens: int = 180_000) -> str`**

Assemble a context string from all documents. Prioritize by label in this order:
1. `Technical Specifications`
2. `Instructions to Bidders`
3. `Financial Template`
4. `Legal / Eligibility Forms`
5. `Annexes / Returnables`
6. `Unknown`

For each document, prepend a header:
```
=== DOCUMENT: {filename} | TYPE: {label} ===
{text}
=== END: {filename} ===
```

Stop adding documents once `estimate_tokens(accumulated_context) > max_tokens`. Log skipped files to `st.session_state["errors"]` with reason `"Token budget exceeded — not included in Q&A context"`.

---

**`answer(question: str, docs: dict) -> dict`**

1. Call `build_context(docs, max_tokens=180_000)`.
2. Call Claude API:

```python
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    system="""You are a tender analysis assistant. Your job is to answer questions about the provided tender documents accurately and concisely.

RULES:
1. Answer using ONLY information found in the provided documents.
2. For every factual claim, cite the source document in brackets: [filename.pdf, Page N] or [filename.docx, Section: X].
3. If the answer is not in the documents, respond exactly: "Not found in the provided tender documents."
4. Do not guess, infer, or use outside knowledge.
5. Structure your answer clearly. Use bullet points for lists.
6. If multiple documents give conflicting information, flag the conflict explicitly.""",
    messages=[
        {
            "role": "user",
            "content": f"TENDER DOCUMENTS:\n\n{context}\n\n---\n\nQUESTION: {question}"
        }
    ]
)
```

3. Return:
```python
{
  "answer": str,                  # response.content[0].text
  "input_tokens": int,            # response.usage.input_tokens
  "output_tokens": int,           # response.usage.output_tokens
  "docs_included": list[str],     # filenames included in context
  "docs_skipped": list[str]       # filenames skipped due to token budget
}
```

---

## MODULE 5 — `core/checklist.py`

### `class ChecklistGenerator`

**`generate(docs: dict) -> list[dict]`**

Scan all documents for mentions of required submission items.

Detection patterns (use `re.findall`, case-insensitive):
```python
patterns = [
    r'annex\s+[\w\-]+',
    r'appendix\s+[\w\-]+',
    r'form\s+[\w\-]+',
    r'attachment\s+[\w\-]+',
    r'returnable\s+[\w\-]+',
    r'(?:bidder shall submit|must be accompanied by|required to submit|submit the following)[^\.\n]{5,80}',
    r'(?:original|copy of|certified copy of)\s+[\w\s]{5,60}(?:certificate|registration|declaration|letter)',
]
```

For each extracted item string:
- Clean and normalize: strip regex artifacts, title-case, max 100 chars.
- `source_file`: file where mention was found.
- `found_in_package`: `True` if any filename in `docs` has a `rapidfuzz.fuzz.partial_ratio` score ≥ 75 against the extracted item string. Otherwise `False`.

Deduplicate across files using `rapidfuzz.fuzz.ratio ≥ 85`.

Return sorted: `found_in_package=False` items first (missing items are the priority).

---

## `app.py` — STREAMLIT UI

### Layout

```
st.set_page_config(page_title="Tender Intelligence System", layout="wide")
```

**Sidebar:**
- `st.text_input("📁 Tender Folder Path", placeholder="/path/to/tender/folder")`
- `st.button("🔍 Analyze Tender")` — triggers full pipeline
- After analysis: show badge summary (files loaded, document type breakdown as `st.metric`)
- `st.expander("⚠️ Processing Errors")` — shows `st.session_state["errors"]` list

**Main area — 5 tabs:**
```python
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Documents",
    "✅ Requirements",
    "💰 Financials",
    "🤖 Q&A",
    "📋 Checklist"
])
```

---

### Tab 1: 📄 Document Overview

- Table: `Filename | Type | Pages/Rows | Confidence | Char Count`
- Color rows by label using `pandas Styler`:
  - Technical Specs → light blue (`#e3f2fd`)
  - Financial → light green (`#e8f5e9`)
  - Legal → light yellow (`#fffde7`)
  - Instructions → light purple (`#f3e5f5`)
  - Annexes → light orange (`#fff3e0`)
  - Unknown → light grey (`#f5f5f5`)

---

### Tab 2: ✅ Requirement Matrix

- `st.dataframe` of RTM. Highlight `Mandatory=True` rows in light red (`#ffebee`).
- Filter: `st.multiselect("Filter by Source File", options=all_filenames)`
- `st.metric("Total Requirements", N)` and `st.metric("Mandatory", M)`
- `st.download_button("⬇ Export CSV", data=df.to_csv(index=False), file_name="requirements.csv")`

---

### Tab 3: 💰 Financial Validation

For each financial file:
- `st.subheader(filename)`
- `st.dataframe(parsed_df)` — show first 50 rows max
- For each flag:
  - `"Math Error"` or `"Currency Inconsistency"` → `st.error(f"🔴 {flag['issue']}: {flag['detail']}")`
  - `"Missing Total"` → `st.warning(f"🟡 {flag['issue']}: {flag['detail']}")`
- If no flags: `st.success("✅ No financial issues detected.")`

---

### Tab 4: 🤖 Q&A Engine

- Check for `ANTHROPIC_API_KEY` in env. If missing: `st.info("Add ANTHROPIC_API_KEY to .env to enable Q&A.")` and `st.stop()`.
- `st.text_area("Ask a question about this tender...", height=80)`
- `st.button("Ask")`
- On submit:
  - Show `st.spinner("Analyzing tender documents...")` while API call runs.
  - Display answer in `st.markdown(answer)` — preserves citation formatting.
  - Show token usage: `st.caption(f"Tokens used: {input_tokens} in / {output_tokens} out")`
  - `st.expander("📂 Documents included in this query")` — list `docs_included` and `docs_skipped`.
- Maintain `st.session_state["qa_history"]` as list of `{question, answer}` dicts.
- Show history below with `st.expander` per prior Q&A.

---

### Tab 5: 📋 Submission Checklist

- Two `st.columns(2)` sections:
  - **Left — ❌ Missing / Not Confirmed** (red background via `st.error` per item)
  - **Right — ✅ Found in Package** (green via `st.success` per item)
- Each item shows: `item_name` + `st.caption(f"Mentioned in: {source_file}")`
- `st.download_button("⬇ Export Checklist", data=df.to_csv(index=False), file_name="checklist.csv")`

---

## SESSION STATE MANAGEMENT

Use `st.session_state` for all pipeline outputs. Never re-run the pipeline on Streamlit rerenders.

```python
# Keys to initialize:
st.session_state.setdefault("docs", None)
st.session_state.setdefault("requirements", None)
st.session_state.setdefault("financial_results", None)
st.session_state.setdefault("checklist", None)
st.session_state.setdefault("errors", [])
st.session_state.setdefault("qa_history", [])
```

Pipeline runs only when `st.button("🔍 Analyze Tender")` is clicked. All tabs read from session state.

---

## ANALYSIS PIPELINE (triggered by button click)

```python
with st.spinner("Loading documents..."):
    progress = st.progress(0)
    ingestor = DocumentIngestor()
    docs = ingestor.load_folder(folder_path)
    progress.progress(20)

with st.spinner("Classifying documents..."):
    docs = ingestor.classify_documents(docs)
    progress.progress(40)

with st.spinner("Extracting requirements..."):
    rm = RequirementMatrix()
    requirements = rm.extract(docs)
    progress.progress(60)

with st.spinner("Parsing financials..."):
    fp = FinancialParser()
    financial_results = fp.parse(docs)
    progress.progress(80)

with st.spinner("Generating checklist..."):
    cg = ChecklistGenerator()
    checklist = cg.generate(docs)
    progress.progress(100)

# Store in session state
st.session_state["docs"] = docs
st.session_state["requirements"] = requirements
st.session_state["financial_results"] = financial_results
st.session_state["checklist"] = checklist
st.success(f"✅ Analysis complete. {len(docs)} files processed.")
```

---

## IMPLEMENTATION RULES

1. **Error handling is mandatory.** Every file I/O and API call must be wrapped in `try/except`. Never crash silently.
2. **Stubs:** If a module cannot be completed without fabricating logic, stub it with `st.info("⚠️ [Module Name]: [what is stubbed and why]")`. Do not write fake regex that won't match real documents.
3. **No org names hardcoded.** The system must work for GIZ, UNOPS, World Bank, EU, private sector tenders equally.
4. **API key:** Load from `.env` using `python-dotenv`. Graceful degradation — only Tab 4 (Q&A) requires the key. All other tabs work without it.
5. **Token budget awareness:** Log to `st.session_state["errors"]` when documents are skipped from Q&A context due to token limits. Never silently drop content.
6. **`tiktoken` estimation:** Use `cl100k_base` as proxy. It's not exact for Claude but within 10% — acceptable for budget gating.
7. **Performance:** `st.progress()` bar during the analysis pipeline. Heavy operations (load_folder, classify_documents) show progress. Never block with no feedback.
8. **No global variables.** All state through `st.session_state` or class instances created inside the pipeline block.

---

## DELIVERABLES

1. All files in the project structure above — complete, runnable code.
2. `requirements.txt` with pinned versions.
3. `.env.example`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```
4. `README.md` with:
   - Setup: `pip install -r requirements.txt`
   - Run: `streamlit run app.py`
   - How to use: paste absolute folder path, click Analyze
   - Known limitations (token budget, regex-based classification accuracy)
   - Cost estimate: ~$0.01–$0.15 per Q&A query depending on tender size

---

## COST AWARENESS NOTE (for developer reference)

`claude-sonnet-4-6` pricing: **$3 / 1M input tokens, $15 / 1M output tokens** — flat rate, no long-context surcharge.

A 300-page tender ≈ 150,000 input tokens per Q&A query ≈ **~$0.45 per query at full context**.

Mitigation already built in: `build_context()` caps at `180,000` tokens and prioritizes by document relevance. For most queries, actual context will be 50,000–100,000 tokens ≈ **$0.15–$0.30 per query**.

For batch analysis (requirement extraction, checklist), Claude API is not called — all regex/fuzzy-match based. Only Tab 4 (Q&A) incurs API cost.