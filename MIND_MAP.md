# DocAI 1.3 — Mind Map

> Conceptual map for AI agents. Shows the system's purpose, functional areas, decision trees, and the relationship between ideas — not code structure.

---

## System Purpose

```
DocAI
└── "Turn document dumps into actionable intelligence — fast."
    ├── WHO: Procurement officers, legal teams, finance reviewers
    ├── WHAT: Upload raw document packages → get structured analysis instantly
    └── HOW: Text extraction → heuristic classification → AI analysis (where needed)
```

---

## Three Operational Modes

```
DocAI
├── 🏗️ TENDER MODE
│   └── "I received a tender package. What do I need to submit and by when?"
│
├── ⚖️ LEGAL MODE
│   └── "I have contracts/NDAs/agreements. What are the risks, obligations, and key dates?"
│
└── 🧾 INVOICE MODE
    └── "I have invoices (PDF or image). Extract all fields into structured data."
```

---

## Tender Mode — Full Concept Tree

```
🏗️ TENDER MODE
│
├── INPUT: Any mix of PDF, DOCX, XLSX, CSV
│
├── STEP 1 — UNDERSTAND THE PACKAGE
│   ├── What files are here?
│   │   └── ingestor.load_files() → DocMeta per file
│   └── What type is each file?
│       └── ingestor.classify_documents()
│           ├── Instructions to Bidders → HOW to submit
│           ├── Technical Specifications → WHAT you must deliver
│           ├── Financial Template → PRICING structure
│           ├── Legal / Eligibility Forms → WHO can bid
│           ├── Annexes / Returnables → DOCUMENTS you must submit
│           └── Unknown → unclassified
│
├── STEP 2 — WHAT MUST I SUBMIT?
│   └── ReturnableDocsExtractor
│       ├── Strategy A (preferred): Ask AI → structured list of required docs
│       └── Strategy B (fallback): Regex scan for "annex", "form", "must submit", etc.
│       └── For each item:
│           ├── Category: Certifications | Forms | Financial | Legal | Technical | Experience
│           ├── Mandatory: yes / no
│           └── Status: Found in package | Missing
│
├── STEP 3 — WHAT ITEMS/LOTS ARE IN THE BoQ?
│   └── ItemsParser
│       └── Per lot: Description | Quantity | Unit Price table
│
├── STEP 4 — DOCUMENT OVERVIEW FOR AI CONTEXT
│   ├── CodemapBuilder → compact ASCII tree of all docs + metadata
│   └── MindMapBuilder → structured conceptual overview of the tender
│   └── Both prepended to Q&A context window
│
└── STEP 5 — ASK ANYTHING
    └── QueryEngine
        ├── Rank docs by relevance to the question (3 signals)
        │   ├── Keyword frequency (normalized by doc length)
        │   ├── Position boost (keywords in first 20% score +50%)
        │   └── Label priority (Tech Specs > ITB > Financial > Legal > Annexes)
        ├── Assemble context (codemap + mindmap header + ranked docs)
        │   ├── Full doc if it fits in budget
        │   ├── Partial truncation if partially fits
        │   └── Skip + log error if no space
        └── Ask OpenRouter (qwen3.5-flash) → cited answer
```

---

## Legal Mode — Full Concept Tree

```
⚖️ LEGAL MODE
│
├── INPUT: Contracts, NDAs, MOUs, policies, license agreements
│
├── CLASSIFY
│   └── LegalAnalyzer.classify_documents()
│       ├── Contract / Agreement
│       ├── NDA / Confidentiality
│       ├── MOU / Letter of Intent
│       ├── Policy / Regulation
│       ├── License Agreement
│       ├── Service Agreement / SOW
│       ├── Corporate / Legal Forms
│       └── Other Legal
│
├── ANALYZE (one AI call for entire batch)
│   └── LegalAnalyzer.analyze()
│       ├── Summary — plain-language overview
│       ├── Parties — who is involved and in what role
│       ├── Key Dates — deadlines, expiry, notice periods
│       ├── Risk Clauses — High / Medium / Low with explanation
│       │   └── e.g. unlimited liability, auto-renewal, IP assignment
│       ├── Obligations — who must do what, mandatory vs optional
│       └── Defined Terms — glossary
│
└── ASK ANYTHING
    └── LegalAnalyzer.answer() → same pattern as Tender Q&A
```

---

## Invoice Mode — Full Concept Tree

```
🧾 INVOICE MODE
│
├── INPUT: PDF, DOCX, XLSX, CSV, PNG, JPG, WEBP
│
├── EXTRACT (one AI call per file)
│   └── InvoiceExtractor.extract()
│       ├── Image files → vision model (base64 encoded)
│       └── Text files → extract text first, then LLM
│
├── STRUCTURED OUTPUT
│   ├── Header: Invoice #, dates, payment terms, PO number
│   ├── Issuer: name, address, contact, tax ID
│   ├── Client: name, address, contact, tax ID
│   ├── Line Items: description, qty, unit, unit price, subtotal, tax
│   ├── Totals: subtotal, discount, tax, total
│   ├── Currency: symbol, code, approximate USD/EUR conversions
│   ├── Bank: account, IBAN, SWIFT, routing
│   └── Notes: free-text
│
└── CONFIDENCE: High | Medium | Low (AI self-assessed)
```

---

## Core Design Decisions

```
DESIGN PHILOSOPHY
│
├── NO VECTOR STORES
│   └── "The whole tender fits in 1M tokens. Retrieval adds complexity without accuracy gain."
│       └── Instead: token-budget context stuffing with relevance ranking
│
├── NO LANGCHAIN
│   └── Direct API calls only — explicit control over prompts and token usage
│
├── HEURISTIC FIRST, AI SECOND
│   ├── Classification: keyword matching (no API cost)
│   ├── Item parsing: structured parsing (no API cost)
│   └── AI only called when human judgment is genuinely needed:
│       ├── Returnable docs extraction (preferred mode)
│       ├── Legal analysis
│       ├── Invoice extraction
│       └── Open-ended Q&A
│
├── GRACEFUL DEGRADATION
│   ├── No API key → Tabs 1-3 still work (heuristic only)
│   ├── API call fails → error shown, other tabs unaffected
│   ├── Returnable docs AI fails → falls back to regex
│   └── Token budget exceeded → partial content included, user notified
│
└── SESSION STATE AS SINGLE SOURCE OF TRUTH
    └── Pipeline runs once on button click
        All tabs read from st.session_state — no recompute on rerender
```

---

## Context Window Strategy

```
Q&A CONTEXT ASSEMBLY (max 180,000 tokens)
│
├── HEADER (always included, counts against budget)
│   ├── Codemap  — full doc listing with types and sizes
│   └── MindMap  — conceptual overview of the tender package
│
├── DOCUMENTS (ordered by relevance to question)
│   ├── Full inclusion: doc fits in remaining budget → include complete
│   ├── Partial inclusion: budget tight → truncate + add [TRUNCATED] notice
│   └── Skip: no space → doc represented only in codemap header
│
└── WHY THIS MATTERS TO THE LLM
    ├── Codemap tells the LLM what docs EXIST (even if not in context)
    ├── Ranking ensures the most relevant doc is read first and in full
    └── Truncation notice tells the LLM it's seeing partial content
```

---

## Data Flow Summary

```
USER UPLOADS FILES
        │
        ▼
DocumentIngestor.load_files()
        │  text, pages, file_type, char_count
        ▼
DocumentIngestor.classify_documents()
        │  + label, confidence
        ▼
        ├──► CodemapBuilder.build()      → codemap string
        ├──► MindMapBuilder.build()      → mindmap string
        ├──► ItemsParser.parse()         → lot/item tables
        └──► ReturnableDocsExtractor.extract()  → required docs list
                        │
                        ▼
              [stored in st.session_state]
                        │
                        ▼
              USER ASKS A QUESTION
                        │
                        ▼
              QueryEngine._rank_docs()  → relevance-ordered doc list
              QueryEngine.build_context() → context string ≤180K tokens
              OpenRouter API call        → cited answer
```

---

## What Each Tab Answers

| Tab | User Question |
|---|---|
| Documents | "What files did I upload and how were they classified?" |
| Returnable Docs | "What do I need to submit and what's missing from my package?" |
| Items | "What line items / lots are in the BoQ?" |
| Q&A | "Any free-form question about the tender content" |
| Legal: Documents | "What legal document types are in this batch?" |
| Legal: Parties | "Who are the parties and what are their roles?" |
| Legal: Key Dates | "What deadlines and notice periods apply?" |
| Legal: Risk | "Which clauses are high-risk and why?" |
| Legal: Obligations | "Who must do what?" |
| Legal: Q&A | "Any free-form question about the legal documents" |
| Invoice: Summary | "Who sent this invoice, to whom, for how much?" |
| Invoice: Line Items | "What are the individual charges?" |
| Invoice: Currency | "What currency and approximate conversions?" |
| Invoice: Bank | "Where do I send payment?" |

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---|---|
| Wrong document classification | Confidence score shown; user can still read full text in Q&A |
| Token budget exceeded | Codemap header preserves doc structure; truncation notice in context |
| AI returns unstructured output (invoice) | `parse_error=True`, raw snippet shown to user |
| OpenRouter API down | `st.error()` per failed call; other tabs unaffected |
| Unsupported file type | Silently skipped; logged to errors |
| Corrupted/password-locked file | Caught in `try/except`; logged to errors; pipeline continues |
| Duplicate returnables | rapidfuzz.ratio ≥ 85 deduplication before storing |
