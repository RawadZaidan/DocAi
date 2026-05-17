# DocAI 1.3 — Code Map

> Structural reference for AI agents. Shows every module, class, method, I/O contract, and dependency edge.

---

## Entry Point

```
app.py
  ├── mode: "🏗️ Tender" | "⚖️ Legal" | "🧾 Invoice"  (st.session_state["active_mode"])
  ├── Sidebar → file_uploader + analyze button per mode
  ├── Pipelines (triggered by button click, guarded by mode)
  │     ├── Tender  → ingestor → items_parser → returnable_docs → codemap/mindmap
  │     ├── Legal   → ingestor → legal_analyzer
  │     └── Invoice → invoice_extractor (per file)
  └── Tabs (read from session state only — no recompute on rerender)
        ├── Tender:  [Documents] [Returnable Docs] [Items] [Q&A]
        ├── Legal:   [Documents] [Parties] [Key Dates] [Risk] [Obligations] [Q&A]
        └── Invoice: [Summary] [Line Items] [Currency] [Bank & Notes]
```

---

## Session State Keys

| Key | Type | Owner |
|---|---|---|
| `active_mode` | str | app.py |
| `docs` | dict\[str, DocMeta\] | Tender pipeline |
| `returnable_docs` | dict | returnable_docs.py |
| `financial_results` | list\[LotResult\] | items_parser.py |
| `codemap` | str | codemap.py |
| `mindmap` | str | mindmap.py |
| `qa_history` | list\[QAEntry\] | query_engine.py |
| `legal_docs` | dict\[str, DocMeta\] | Legal pipeline |
| `legal_result` | dict | legal_analyzer.py |
| `legal_qa_history` | list\[QAEntry\] | legal_analyzer.py |
| `invoice_results` | list\[InvoiceResult\] | invoice_extractor.py |
| `errors` | list\[ErrorEntry\] | all modules |

---

## DocMeta Shape (shared by Tender + Legal)

```python
{
  "text": str,           # Full extracted text
  "pages": list[str],    # Per-page (PDF) or per-sheet (XLSX) text segments
  "file_type": str,      # "pdf" | "docx" | "xlsx" | "csv"
  "char_count": int,
  "label": str,          # Set by classify_documents()
  "confidence": float,   # 0.0–1.0
  "path": str,           # Original file path or filename
  "raw_bytes": bytes,    # Only when loaded via load_files() (upload mode)
}
```

---

## Module Dependency Graph

```
app.py
  ├── core.ingestor        (DocumentIngestor)
  ├── core.items_parser    (ItemsParser)           ← depends on ingestor output
  ├── core.returnable_docs (ReturnableDocsExtractor) ← depends on ingestor output
  ├── core.query_engine    (QueryEngine)            ← depends on ingestor output + codemap/mindmap
  ├── core.codemap         (CodemapBuilder)         ← depends on ingestor output
  ├── core.mindmap         (MindMapBuilder)         ← depends on ingestor output
  ├── core.legal_analyzer  (LegalAnalyzer)          ← depends on ingestor output; calls OpenRouter
  └── core.invoice_extractor (InvoiceExtractor)     ← standalone; calls OpenRouter vision model
```

Unused legacy modules (kept, not wired to app.py):
- `core/requirement_matrix.py` — `RequirementMatrix`
- `core/financial_parser.py`   — `FinancialParser`
- `core/checklist.py`          — `ChecklistGenerator`

---

## `core/ingestor.py`

**Class:** `DocumentIngestor`

| Method | Signature | Returns |
|---|---|---|
| `load_folder` | `(folder_path: str) -> dict` | DocMeta dict keyed by filename |
| `load_files` | `(uploaded_files: list) -> dict` | DocMeta dict (Streamlit UploadedFile list) |
| `classify_documents` | `(docs: dict) -> dict` | Same dict with `label` + `confidence` filled |
| `_extract` | `(fpath, ext) -> (text, pages)` | Internal — routes by extension |
| `_extract_fileobj` | `(fileobj, ext) -> (text, pages)` | Internal — for BytesIO objects |
| `_extract_pdf` | `(source) -> (text, pages)` | pypdf.PdfReader, page separator `\n--- Page N ---\n` |
| `_extract_docx` | `(source) -> (text, pages)` | python-docx, headings prefixed `##` |
| `_extract_xlsx` | `(source) -> (text, pages)` | openpyxl → pandas → markdown per sheet |
| `_extract_csv` | `(source) -> (text, pages)` | pandas → markdown |

**Label taxonomy (Tender):**

| Label | Trigger keywords |
|---|---|
| `Instructions to Bidders` | itb, rfq, rfp, submission deadline, bidding procedures |
| `Technical Specifications` | scope of work, deliverable, shall, must, technical criteria |
| `Financial Template` | boq, unit price, financial offer, price schedule |
| `Legal / Eligibility Forms` | eligibility, declaration, certificate, authorized signatory |
| `Annexes / Returnables` | annex, appendix, form, returnable, attachment |
| `Unknown` | default |

Confidence = matched_keywords / total_keywords_in_category (first 3000 + last 1000 chars).

---

## `core/query_engine.py`

**Class:** `QueryEngine`  
**Backend:** OpenRouter API (`qwen/qwen3.5-flash-02-23`) via `openai` SDK  
**Tokenizer:** `tiktoken cl100k_base` (proxy, ~10% error margin)

| Method | Signature | Returns |
|---|---|---|
| `estimate_tokens` | `(text: str) -> int` | Token count |
| `_rank_docs` | `(question, docs) -> list[str]` | Filenames sorted by relevance score |
| `build_context` | `(docs, max_tokens=180_000, question="", codemap="", mindmap="") -> tuple` | `(context_str, included_list, skipped_list)` |
| `answer` | `(question, docs, codemap="", mindmap="") -> dict` | QAResult dict |

**`_rank_docs` scoring (3 signals):**
1. Keyword frequency normalized by `log(word_count)` — removes long-doc bias
2. Position boost (+50%) for keywords in first 20% of doc
3. Label-priority boost: Technical Specs > ITB > Financial > Legal > Annexes > Unknown

**`build_context` strategy:**
- Prepends codemap + mindmap header (counts against budget)
- Ranks docs by relevance if `question` provided, else by label priority
- Partial truncation when a doc is too large to fit fully — appends `[... TRUNCATED ...]` notice
- Docs that can't fit even partially: logged to `st.session_state["errors"]`

**QAResult shape:**
```python
{
  "answer": str,
  "input_tokens": int,
  "output_tokens": int,
  "docs_included": list[str],
  "docs_skipped": list[str],
  "corpus_coverage": float,   # len(included) / len(docs)
}
```

---

## `core/items_parser.py`

**Class:** `ItemsParser`

Parses line items from financial documents, grouping by lot.

| Method | Returns |
|---|---|
| `parse(docs)` | `list[LotResult]` |

**LotResult shape:**
```python
{
  "filename": str,
  "lot_name": str,
  "items": pd.DataFrame,   # columns: Description, Quantity, Unit Price
  "error": str | None,
}
```

---

## `core/returnable_docs.py`

**Class:** `ReturnableDocsExtractor`

Identifies required submission documents (returnables) from tender text.  
Attempts AI extraction via OpenRouter first; falls back to regex.

| Method | Returns |
|---|---|
| `extract(docs)` | dict with `items`, `method`, `docs_included`, `docs_skipped`, `error` |
| `to_dataframe(rd)` | `pd.DataFrame` — columns: Doc Name, Category, Mandatory, Status, Source |

**Item categories:** Certifications, Forms, Financial Documents, Legal / Corporate, Technical Documents, Experience / References, Other

**Item shape:**
```python
{
  "doc_name": str,
  "category": str,
  "mandatory": bool,
  "description": str,
  "source_file": str,
  "found_in_package": bool,
}
```

---

## `core/legal_analyzer.py`

**Class:** `LegalAnalyzer`  
**Backend:** OpenRouter API

| Method | Returns |
|---|---|
| `classify_documents(docs)` | docs dict with legal labels + confidence |
| `analyze(docs)` | LegalResult dict |
| `answer(question, docs)` | QAResult dict |

**Legal label taxonomy:**
Contract / Agreement, NDA / Confidentiality, MOU / Letter of Intent,
Policy / Regulation, License Agreement, Service Agreement / SOW,
Corporate / Legal Forms, Other Legal

**LegalResult shape:**
```python
{
  "summary": str,
  "parties": list[{"name", "role", "description"}],
  "key_dates": list[{"event", "date", "source"}],
  "risk_clauses": list[{"type", "risk_level", "clause_excerpt", "explanation", "source"}],
  "obligations": list[{"party", "obligation", "mandatory", "source"}],
  "defined_terms": list[{"term", "definition"}],
  "input_tokens": int,
  "output_tokens": int,
  "error": str | None,
}
```

---

## `core/invoice_extractor.py`

**Class:** `InvoiceExtractor`  
**Backend:** OpenRouter API (vision model for images; text model for PDF/DOCX/XLSX)

| Method | Returns |
|---|---|
| `extract(file_obj, filename, ext)` | InvoiceResult dict |

**InvoiceResult shape:**
```python
{
  "filename": str,
  "invoice_number": str,
  "invoice_date": str,
  "due_date": str,
  "payment_terms": str,
  "validity": str,
  "po_number": str,
  "issuer": {"name", "address", "city", "country", "phone", "email", "website", "tax_id", "registration"},
  "client": {"name", "contact_person", "address", "city", "country", "phone", "email", "tax_id"},
  "line_items": list[{"description", "quantity", "unit", "unit_price", "subtotal", "tax_rate"}],
  "subtotal": float,
  "discount": float,
  "tax_amount": float,
  "total": float,
  "currency": str,
  "currency_symbol": str,
  "currency_conversions": {"total_usd", "total_eur", "usd_rate", "eur_rate", "rate_note"},
  "bank_details": {"bank_name", "account_number", "iban", "swift", "routing"},
  "notes": str,
  "confidence": "High" | "Medium" | "Low",
  "input_tokens": int,
  "error": str | None,
  "parse_error": bool,
  "raw_snippet": str,
}
```

---

## `core/codemap.py`

**Class:** `CodemapBuilder`

| Method | Returns |
|---|---|
| `build(docs)` | `str` — compact ASCII tree of all documents with metadata |

Output prepended to every Q&A context window. Gives the LLM a structural overview of all docs even when some are truncated.

---

## `core/mindmap.py`

**Class:** `MindMapBuilder`

| Method | Returns |
|---|---|
| `build(docs)` | `str` — structured text mind map of the tender package |

Output prepended alongside codemap in Q&A context.

---

## Legacy Modules (not wired to app.py)

| Module | Class | Status |
|---|---|---|
| `core/requirement_matrix.py` | `RequirementMatrix` | Regex-based — unused |
| `core/financial_parser.py` | `FinancialParser` | Replaced by `ItemsParser` |
| `core/checklist.py` | `ChecklistGenerator` | Replaced by `ReturnableDocsExtractor` |

---

## External Dependencies

| Package | Role |
|---|---|
| `streamlit` | UI framework |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `openpyxl` | XLSX parsing |
| `pandas` | Tabular data, markdown output |
| `rapidfuzz` | Fuzzy string matching (dedup, column detection) |
| `openai` | OpenRouter API client (drop-in) |
| `tiktoken` | Token estimation (cl100k_base proxy) |
| `python-dotenv` | `.env` loading |

**Env vars required:**
- `OPENROUTER_API_KEY` — needed for Q&A, Legal analysis, Invoice extraction
- `OPENROUTER_BASE_URL` = `https://openrouter.ai/api/v1` (hardcoded in query_engine.py)

---

## Error Handling Contract

- All file I/O is wrapped in `try/except`
- Failures append `{"file": str, "error": str}` to `st.session_state["errors"]`
- Errors surface in sidebar expander: `⚠️ Processing Errors (N)`
- API failures surface as `st.error()` in the relevant tab
- Token budget exhaustion: file added to `docs_skipped`, logged to errors
