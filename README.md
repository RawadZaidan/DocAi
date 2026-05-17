# Tender Intelligence System (TIS)

Analyze professional tender packages (PDF, DOCX, XLSX, CSV) using Claude's 1M-token context window. Surfaces document classification, returnable document identification, items/lot parsing, and a full-context Q&A engine.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

## Run

```bash
streamlit run app.py
```

## How to Use

1. Launch the app with the command above.
2. In the sidebar, paste the **absolute path** to the folder containing your tender documents.
3. Click **🔍 Analyze Tender**.
4. Browse results across 4 tabs: Documents, Returnable Docs, Items, Q&A.

Only the **Q&A tab** requires an `OPENROUTER_API_KEY`. All other tabs run entirely offline.

## Supported Formats

| Format | Extraction method |
|--------|------------------|
| PDF    | `pypdf` page-by-page text extraction |
| DOCX   | `python-docx` paragraphs + tables |
| XLSX   | `openpyxl` → pandas → markdown tables |
| CSV    | `pandas` → markdown table |

## Known Limitations

- **Classification accuracy** depends on keyword presence in the first 3,000 + last 1,000 characters of each document. Short or scanned PDFs may be classified as `Unknown`.
- **Returnable doc extraction** uses AI when an `OPENROUTER_API_KEY` is present, otherwise falls back to regex patterns. AI mode is significantly more accurate.
- **Items/lot parsing** assumes tabular data with recognisable column headers. Non-standard or merged-cell layouts may not parse correctly.
- **Token budget** for Q&A is capped at 180,000 tokens per query. Very large tender packages will have lower-priority documents excluded from context; excluded files are listed in the sidebar error log.
- **Scanned PDFs** (image-only) will yield no extracted text. OCR is not included in this version.

## Cost Estimate (Q&A Tab)

| Tender size | Approximate cost per query |
|-------------|---------------------------|
| Small (< 50 pages) | ~$0.01–$0.05 |
| Medium (50–150 pages) | ~$0.05–$0.15 |
| Large (150–300 pages) | ~$0.15–$0.45 |

Pricing varies by model via OpenRouter. Document classification, returnable doc extraction (regex fallback), and items parsing incur **no API cost**.
