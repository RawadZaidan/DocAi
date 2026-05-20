# DocAI

An AI-powered document intelligence platform built with Streamlit. Upload any batch of professional documents and get structured extraction, classification, risk analysis, and Q&A — across three specialized modes.

---

## Modes

| Mode | What it does |
|------|-------------|
| **Tender** | Classifies tender documents, identifies returnable/submission items, parses BoQ line items by lot, and answers natural-language questions grounded in the package |
| **Legal** | Classifies legal documents, extracts parties, key dates, obligations, and risk clauses with severity ratings, plus a legal Q&A engine |
| **Invoice** | Extracts structured invoice data (line items, totals, issuer, client, bank details, shipping, T&C, notes) from PDF, image, or spreadsheet files |

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd DocAI-1.3
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies (`requirements.txt`):

```
streamlit==1.35.0
pypdf==4.2.0
python-docx==1.1.2
openpyxl==3.1.2
pandas==2.2.2
tabulate==0.9.0
rapidfuzz==3.9.3
openai==1.35.0
python-dotenv==1.0.1
tiktoken==0.7.0
```

**Optional but recommended** — install for full PDF vision support (scanned PDFs):

```bash
pip install pdfplumber pymupdf pdf2image Pillow
```

> `pdf2image` also requires [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) on Windows. Add the Poppler `bin/` folder to your PATH.

### 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and add your [OpenRouter](https://openrouter.ai) key:

```
OPENROUTER_API_KEY=sk-or-...
```

All three modes call OpenRouter. No Anthropic key is needed.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Select a mode from the sidebar radio buttons (**Tender / Legal / Invoice**).
2. Upload your files using the sidebar uploader.
3. Click the **Analyze / Extract** button.
4. Browse results across tabs in the main panel.

### Invoice Scanner (standalone CLI)

You can also run the invoice extractor directly from the command line:

```bash
# Single file
python invoice_scanner.py path/to/invoice.pdf

# Batch (entire folder)
python invoice_scanner.py ./invoices/

# Image input
python invoice_scanner.py scan.jpg
```

Results are saved to `invoice_results.json` in the working directory.

---

## Capabilities

### Tender Mode

- **Document classification** — labels each file as Technical Specs, Financial Template, Legal/Eligibility Forms, Instructions to Bidders, Annexes/Returnables, or Unknown, with a confidence score.
- **Returnable document detection** — identifies submission items required by the tender (forms, certificates, financial statements, etc.) and flags which are present vs. missing in the uploaded package. AI-powered when `OPENROUTER_API_KEY` is set; regex fallback otherwise.
- **BoQ / items parsing** — detects price schedule tables, groups line items by lot, and exports them as CSV.
- **Codemap & mind map** — generates a structured codemap and hierarchical mind map of the tender package.
- **Q&A engine** — answers free-text questions strictly grounded in the loaded documents, with source attribution and corpus coverage metrics.

### Legal Mode

- **Document classification** — categorises files as Contract/Agreement, NDA, MOU, Policy/Regulation, License Agreement, Service Agreement, Corporate Forms, or Other Legal.
- **Party extraction** — identifies all named parties and their roles.
- **Key dates** — extracts deadlines, effective dates, expiry dates, and renewal windows.
- **Risk analysis** — flags risky clauses (indemnification, liability caps, termination, penalties) with High / Medium / Low severity and plain-language explanations.
- **Obligations** — lists every obligation per party, marking mandatory vs. optional, with the source document.
- **Defined terms** — extracts the definitions section.
- **Legal Q&A** — answers questions grounded in the uploaded documents.

### Invoice Mode

- **Hybrid PDF pipeline** — tries native text extraction first (`pdfplumber` → `fitz` fallback); falls back to full-page vision rendering (`pdf2image` → `fitz`) for scanned/image PDFs. Typical processing time: 5–12 seconds per invoice.
- **Multi-language support** — handles Arabic, French, English, and mixed-language documents.
- **Structured extraction** — invoice number, dates, issuer, client, line items (qty, unit price, subtotal, tax rate), totals, currency, currency conversions (USD/EUR estimates), bank details (IBAN, SWIFT), shipping/logistics fields, notes, and terms & conditions.
- **Multi-page merging** — for multi-page invoices, line items are concatenated across pages and totals are taken from the last page.
- **Batch processing** — upload multiple invoices at once; a summary table compares all results side-by-side.
- **CSV export** — line items for each invoice can be exported.

### Models used

| Feature | Model |
|---------|-------|
| Invoice extraction (vision + text) | `google/gemini-2.5-flash-lite` via OpenRouter |
| Legal analysis, Tender Q&A, Returnable docs | `qwen/qwen3.5-flash-02-23` via OpenRouter |

---

## Supported File Formats

| Format | Extraction method |
|--------|-----------------|
| PDF (text-layer) | `pdfplumber` → `fitz` (pymupdf) fallback |
| PDF (scanned/image) | `pdf2image` (Poppler) → `fitz` fallback → vision LLM |
| DOCX | `python-docx` paragraphs + tables |
| XLSX | `openpyxl` → pandas → markdown |
| CSV | `pandas` → markdown |
| PNG / JPG / WEBP | Vision LLM directly |

---

## Limitations

- **Scanned PDFs without optional dependencies** — if `pdfplumber`, `pymupdf`, and `pdf2image` are not installed, scanned (image-only) PDFs cannot be rendered and will return empty results.
- **Invoice extraction accuracy** — results depend on document quality and layout. Handwritten invoices, very low-resolution scans, or heavily stylised layouts may yield lower confidence.
- **Currency conversions** — USD/EUR equivalents in the invoice mode are approximate estimates from the model's training data, not live exchange rates.
- **Token budget** — the Tender Q&A engine caps context at ~180,000 tokens per query. Very large document sets will have lower-priority files excluded; excluded files are listed in the UI.
- **Tender/Legal classification** — uses keyword matching on the first and last portions of each document. Very short documents or those with non-standard language may be classified as Unknown/Other.
- **BoQ parsing** — assumes tabular data with recognisable column headers (Description, Quantity, Unit Price). Non-standard or merged-cell layouts may not parse correctly.
- **No offline mode for AI features** — Legal analysis, Invoice extraction, Returnable doc detection (AI mode), and Q&A all require a valid `OPENROUTER_API_KEY` and an internet connection.
- **Language** — the UI is in English. Document content in Arabic and French is handled by the models, but the interface labels are English-only.
