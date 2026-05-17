import os
import io
import re
import base64
import json
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEXT_MODEL = "qwen/qwen3.5-flash-02-23"
VISION_MODEL = "anthropic/claude-sonnet-4-6"

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif"}

# Confidence threshold: below this score the local extractor defers to AI
_AI_FALLBACK_THRESHOLD = 4

EXTRACTION_PROMPT = """You are an expert invoice data extraction system. Extract ALL data visible in the invoice.

Return ONLY raw JSON (no markdown fences, no explanation). Use this exact structure:
{
  "invoice_number": null,
  "invoice_date": null,
  "due_date": null,
  "payment_terms": null,
  "validity": null,
  "po_number": null,
  "issuer": {
    "name": null,
    "address": null,
    "city": null,
    "country": null,
    "phone": null,
    "email": null,
    "website": null,
    "tax_id": null,
    "registration": null
  },
  "client": {
    "name": null,
    "address": null,
    "city": null,
    "country": null,
    "phone": null,
    "email": null,
    "contact_person": null,
    "tax_id": null
  },
  "line_items": [
    {
      "description": null,
      "quantity": null,
      "unit": null,
      "unit_price": null,
      "subtotal": null,
      "tax_rate": null
    }
  ],
  "subtotal": null,
  "tax_amount": null,
  "discount": null,
  "total": null,
  "currency": null,
  "currency_symbol": null,
  "currency_conversions": {
    "usd_rate": null,
    "eur_rate": null,
    "total_usd": null,
    "total_eur": null,
    "rate_note": "Approximate rates based on model knowledge — verify before use"
  },
  "bank_details": {
    "bank_name": null,
    "account_number": null,
    "iban": null,
    "swift": null,
    "routing": null
  },
  "notes": null,
  "confidence": "High"
}

Rules:
- Use null for any field not present in the invoice.
- Numbers must be numeric (not strings): quantity, unit_price, subtotal, total, tax_amount, etc.
- For currency_conversions: if the invoice is NOT in USD, provide approximate USD and EUR equivalents using your knowledge of exchange rates. If already USD, set usd_rate to 1.0.
- confidence: High if most fields extracted cleanly, Medium if some ambiguity, Low if document is unclear."""


# ---------------------------------------------------------------------------
# Fast local extractor — pure regex, no network call
# ---------------------------------------------------------------------------

class _LocalExtractor:
    _INV_NUM = re.compile(
        r'(?:invoice\s*(?:no\.?|number|#)|inv\.?\s*(?:no\.?|#))'
        r'[\s:]*([A-Z0-9][A-Z0-9\-\/\.]{1,25})',
        re.IGNORECASE,
    )
    _DATE = re.compile(
        r'(?:invoice\s+date|date\s+issued|date\s+of\s+invoice|date)[:\s]+'
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'
        r'|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'
        r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{1,2},?\s+\d{4}'
        r'|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4})',
        re.IGNORECASE,
    )
    _DUE = re.compile(
        r'(?:due\s+date|payment\s+due|due\s+by|pay\s+by)[:\s]+'
        r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'
        r'|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'
        r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE,
    )
    _TOTAL = re.compile(
        r'(?:grand\s+total|total\s+amount\s+due|total\s+due|amount\s+due'
        r'|total\s+payable|net\s+payable|balance\s+due|total)[:\s]*'
        r'([€$£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )
    _SUBTOTAL = re.compile(
        r'(?:subtotal|sub\s*total|net\s+amount)[:\s]*([€$£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )
    _TAX = re.compile(
        r'(?:vat|tax|gst|hst|pst)[:\s]*(?:\d+\.?\d*%\s*)?([€$£¥₹]?\s*[\d,]+\.?\d*)',
        re.IGNORECASE,
    )
    _PO = re.compile(
        r'(?:p\.?o\.?\s*(?:no\.?|number|#)|purchase\s+order\s*(?:no\.?|#)?)'
        r'[\s:]*([A-Z0-9\-\/\.]{2,25})',
        re.IGNORECASE,
    )
    _TERMS = re.compile(
        r'(?:payment\s+terms?|terms?)[:\s]+(net\s*\d+|due\s+on\s+receipt|immediate|[^\n]{3,40})',
        re.IGNORECASE,
    )
    _CURRENCY_SYM = re.compile(r'([€$£¥₹])\s*[\d,]+\.?\d')
    _CURRENCY_CODE = re.compile(
        r'\b(USD|EUR|GBP|CHF|JPY|AED|SAR|LBP|LYD|EGP|QAR|KWD|TRY|INR|CAD|AUD|SGD)\b'
    )
    _IBAN = re.compile(r'\bIBAN[:\s]*([A-Z]{2}\d{2}[\w\s]{10,30})', re.IGNORECASE)
    _SWIFT = re.compile(r'\b(?:SWIFT|BIC)[:\s]*([A-Z]{6}[A-Z0-9]{2,5})', re.IGNORECASE)
    _BANK = re.compile(r'(?:bank\s+name|bank)[:\s]+([^\n,]{3,50})', re.IGNORECASE)
    _ACCOUNT = re.compile(r'(?:account\s*(?:no\.?|number|#))[:\s]*([A-Z0-9\-]{4,30})', re.IGNORECASE)
    _EMAIL = re.compile(r'[\w.\-+]+@[\w\-]+\.[\w.]{2,}')
    _PHONE = re.compile(r'(?:tel|phone|mob|fax)?[\s:]*(\+?[\d\s\-().]{7,20})')
    # Markdown table row: | val | val | ...
    _MD_TABLE_ROW = re.compile(r'^\|(.+)\|$', re.MULTILINE)

    _SYM_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
    _CODE_TO_SYM = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}

    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        result = {
            "invoice_number": None, "invoice_date": None, "due_date": None,
            "payment_terms": None, "validity": None, "po_number": None,
            "subtotal": None, "tax_amount": None, "discount": None,
            "total": None, "currency": None, "currency_symbol": None, "notes": None,
            "issuer": {k: None for k in ["name", "address", "city", "country", "phone", "email", "website", "tax_id", "registration"]},
            "client": {k: None for k in ["name", "address", "city", "country", "phone", "email", "contact_person", "tax_id"]},
            "line_items": [],
            "bank_details": {k: None for k in ["bank_name", "account_number", "iban", "swift", "routing"]},
            "currency_conversions": {"usd_rate": None, "eur_rate": None, "total_usd": None, "total_eur": None, "rate_note": None},
        }

        m = self._INV_NUM.search(text)
        if m:
            result["invoice_number"] = m.group(1).strip()

        m = self._DATE.search(text)
        if m:
            result["invoice_date"] = m.group(1).strip()

        m = self._DUE.search(text)
        if m:
            result["due_date"] = m.group(1).strip()

        m = self._PO.search(text)
        if m:
            result["po_number"] = m.group(1).strip()

        m = self._TERMS.search(text)
        if m:
            result["payment_terms"] = m.group(1).strip()[:60]

        result["total"] = self._amount(self._TOTAL, text)
        result["subtotal"] = self._amount(self._SUBTOTAL, text)
        result["tax_amount"] = self._amount(self._TAX, text)

        sym_m = self._CURRENCY_SYM.search(text)
        code_m = self._CURRENCY_CODE.search(text)
        if sym_m:
            result["currency_symbol"] = sym_m.group(1)
            result["currency"] = self._SYM_TO_CODE.get(sym_m.group(1), "Unknown")
        if code_m:
            result["currency"] = code_m.group(1)
            if not result["currency_symbol"]:
                result["currency_symbol"] = self._CODE_TO_SYM.get(code_m.group(1), "")

        m = self._IBAN.search(text)
        if m:
            result["bank_details"]["iban"] = re.sub(r"\s+", "", m.group(1))
        m = self._SWIFT.search(text)
        if m:
            result["bank_details"]["swift"] = m.group(1).strip()
        m = self._BANK.search(text)
        if m:
            result["bank_details"]["bank_name"] = m.group(1).strip()
        m = self._ACCOUNT.search(text)
        if m:
            result["bank_details"]["account_number"] = m.group(1).strip()

        result["line_items"] = self._extract_line_items(text)

        # Confidence score
        score = (
            bool(result["invoice_number"]) * 2
            + bool(result["total"]) * 2
            + bool(result["invoice_date"])
            + bool(result["currency"])
            + bool(result["subtotal"])
            + (1 if result["line_items"] else 0)
        )
        result["confidence"] = "High" if score >= 6 else "Medium" if score >= _AI_FALLBACK_THRESHOLD else "Low"
        result["_score"] = score
        return result

    def _amount(self, pattern: re.Pattern, text: str):
        m = pattern.search(text)
        if not m:
            return None
        raw = re.sub(r"[€$£¥₹\s,]", "", m.group(1))
        try:
            return float(raw)
        except ValueError:
            return None

    def _extract_line_items(self, text: str) -> list:
        """Parse markdown table rows (produced by the XLSX/CSV ingestor) into line items."""
        rows = self._MD_TABLE_ROW.findall(text)
        if len(rows) < 2:
            return []

        header = [c.strip().lower() for c in rows[0].split("|")]
        items = []
        for row in rows[2:]:  # skip separator row
            cells = [c.strip() for c in row.split("|")]
            if len(cells) < len(header):
                continue
            item = {}
            for i, col in enumerate(header):
                val = cells[i] if i < len(cells) else ""
                if not val or val in ("-", "nan", "none", ""):
                    continue
                if any(k in col for k in ("desc", "item", "particular", "service", "work")):
                    item["description"] = val
                elif any(k in col for k in ("qty", "quant", "count")):
                    item["quantity"] = self._to_num(val)
                elif any(k in col for k in ("unit", "uom")):
                    item["unit"] = val
                elif any(k in col for k in ("unit price", "rate", "unit cost")):
                    item["unit_price"] = self._to_num(val)
                elif any(k in col for k in ("total", "amount", "subtotal", "extended")):
                    item["subtotal"] = self._to_num(val)
                elif "tax" in col:
                    item["tax_rate"] = val
            if item.get("description"):
                items.append(item)
        return items

    @staticmethod
    def _to_num(val: str):
        try:
            return float(re.sub(r"[^\d.]", "", val.replace(",", "")))
        except (ValueError, AttributeError):
            return None


_local = _LocalExtractor()


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class InvoiceExtractor:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )

    def extract(self, file_obj: io.BytesIO, filename: str, file_type: str) -> dict:
        try:
            ft = file_type.lower().lstrip(".")
            if ft in IMAGE_TYPES:
                return self._from_image(file_obj, filename, ft)
            else:
                return self._from_text(file_obj, filename, ft)
        except Exception as e:
            return {
                "filename": filename,
                "error": str(e),
                "invoice_number": None,
                "issuer": {}, "client": {}, "line_items": [],
                "total": None, "currency": None,
                "input_tokens": 0, "output_tokens": 0,
            }

    def _from_text(self, file_obj: io.BytesIO, filename: str, file_type: str) -> dict:
        from core.ingestor import DocumentIngestor
        file_obj.seek(0)
        raw_bytes = file_obj.read()
        file_obj.seek(0)
        text, _ = DocumentIngestor()._extract_fileobj(io.BytesIO(raw_bytes), f".{file_type}")

        # Fast path: local regex extraction — returns in ms
        local = _local.extract(text)
        score = local.pop("_score", 0)

        if score >= _AI_FALLBACK_THRESHOLD:
            local["filename"] = filename
            local["input_tokens"] = 0
            local["output_tokens"] = 0
            return local

        # If extracted text is too sparse (scanned/image PDF), use vision model with raw PDF
        meaningful_chars = len(text.strip().replace("\n", "").replace("-", "").replace(" ", ""))
        if file_type == "pdf" and meaningful_chars < 200:
            return self._from_pdf_vision(io.BytesIO(raw_bytes), filename)

        # Slow path: AI fallback for low-confidence text documents
        response = self.client.chat.completions.create(
            model=TEXT_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract all invoice data from this document:\n\n{text[:30000]}"},
            ],
        )
        return self._parse(response, filename)

    def _from_pdf_vision(self, file_obj: io.BytesIO, filename: str) -> dict:
        """Extract images from a scanned PDF page and send to the vision model."""
        import pypdf

        file_obj.seek(0)
        reader = pypdf.PdfReader(file_obj)

        # Collect images from the first few pages (covers multi-page invoice headers)
        content_blocks = []
        for page in reader.pages[:4]:
            try:
                for img in page.images:
                    raw = img.data
                    name = (img.name or "").lower()
                    mime = "image/png" if name.endswith(".png") else "image/jpeg"
                    b64 = base64.standard_b64encode(raw).decode("utf-8")
                    content_blocks.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    )
            except Exception:
                continue
            if len(content_blocks) >= 6:  # cap at 6 images to stay within token limits
                break

        if not content_blocks:
            # No embedded images — PDF may be corrupted or purely vector; return empty shell
            return {
                "filename": filename,
                "confidence": "Low",
                "invoice_number": None, "invoice_date": None, "due_date": None,
                "payment_terms": None, "validity": None, "po_number": None,
                "issuer": {k: None for k in ["name", "address", "city", "country", "phone", "email", "website", "tax_id", "registration"]},
                "client": {k: None for k in ["name", "address", "city", "country", "phone", "email", "contact_person", "tax_id"]},
                "line_items": [],
                "subtotal": None, "tax_amount": None, "discount": None,
                "total": None, "currency": None, "currency_symbol": None,
                "currency_conversions": {"usd_rate": None, "eur_rate": None, "total_usd": None, "total_eur": None, "rate_note": None},
                "bank_details": {k: None for k in ["bank_name", "account_number", "iban", "swift", "routing"]},
                "notes": "Could not extract text or images from this PDF.",
                "input_tokens": 0, "output_tokens": 0,
            }

        content_blocks.append(
            {"type": "text", "text": EXTRACTION_PROMPT + "\n\nExtract all invoice data from these invoice page images."}
        )

        response = self.client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": content_blocks}],
        )
        return self._parse(response, filename)

    def _from_image(self, file_obj: io.BytesIO, filename: str, file_type: str) -> dict:
        file_obj.seek(0)
        b64 = base64.standard_b64encode(file_obj.read()).decode("utf-8")
        mime = "image/jpeg" if file_type == "jpg" else f"image/{file_type}"

        response = self.client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": EXTRACTION_PROMPT + "\n\nExtract all invoice data from this image."},
                    ],
                }
            ],
        )
        return self._parse(response, filename)

    def _parse(self, response, filename: str) -> dict:
        raw = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "parse_error": "Could not parse AI response as JSON",
                "raw_snippet": raw[:400],
                "line_items": [],
            }

        data["filename"] = filename
        data["input_tokens"] = input_tokens
        data["output_tokens"] = output_tokens
        return data
