# Document Intelligence Agent

An AI-powered document analysis agent that ingests any folder of professional documents (PDF, DOCX, XLSX, CSV) and surfaces structured intelligence: automatic classification, requirement extraction, financial validation, returnable item identification, and a full-context Q&A engine — all powered by Claude's 1M-token context window.

Designed for any document-heavy workflow: procurement packages, tender dossiers, contract bundles, compliance packs, project deliverables, or due-diligence sets.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Run

```bash
streamlit run app.py
```

## How to Use

1. Launch the app with the command above.
2. In the sidebar, paste the **absolute path** to the folder containing your documents.
3. Click **Analyze**.
4. Browse results across tabs: Documents, Requirements, Financials, Q&A, Checklist.

Only the **Q&A tab** requires an `ANTHROPIC_API_KEY`. All other tabs run entirely offline.

## Supported Formats

| Format | Extraction method |
|--------|------------------|
| PDF    | `pypdf` page-by-page text extraction |
| DOCX   | `python-docx` paragraphs + tables |
| XLSX   | `openpyxl` → pandas → markdown tables |
| CSV    | `pandas` → markdown table |

## What the Agent Does

| Tab | What it surfaces |
|-----|-----------------|
| **Documents** | Classifies each file by type (specs, financials, legal, annexes) with confidence score |
| **Requirements** | Extracts every SHALL / MUST / REQUIRED sentence into a traceable requirement matrix |
| **Financials** | Detects BoQ/price-schedule tables, validates row math, flags currency inconsistencies |
| **Q&A** | Answers any natural-language question grounded strictly in the loaded documents, with citations |
| **Checklist** | Identifies required submission items and flags which are present vs. missing in the package |

## Known Limitations

- **Classification accuracy** depends on keyword presence in the first 3,000 + last 1,000 characters of each document. Short or scanned PDFs may be classified as `Unknown`.
- **Requirement extraction** is regex-based; it catches explicit obligation language but will miss implied or paraphrased requirements.
- **Financial parsing** assumes tabular data with recognisable column headers. Non-standard or merged-cell layouts may not parse correctly.
- **Token budget** for Q&A is capped at 180,000 tokens per query. Very large document sets will have lower-priority files excluded from context; excluded files are listed in the sidebar error log.
- **Scanned PDFs** (image-only) yield no extracted text. OCR is not included in this version.

## Cost Estimate (Q&A Tab)

| Document set size | Approximate cost per query |
|-------------------|---------------------------|
| Small (< 50 pages) | ~$0.01–$0.05 |
| Medium (50–150 pages) | ~$0.05–$0.15 |
| Large (150–300 pages) | ~$0.15–$0.45 |

Powered by `claude-sonnet-4-6` ($3 / 1M input tokens). Document classification, requirement extraction, financial parsing, and checklist generation incur **no API cost**.
