import io
import base64
import json
import time
from openai import OpenAI
from core.utils import strip_json_fences


def _timed(label: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"[invoice_extractor] {label}: {elapsed:.2f}s")
            return result
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VISION_MODEL = "google/gemini-2.5-flash-lite"

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif"}

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
  "shipping": {
    "shipping_mode": null,
    "incoterms": null,
    "port_of_loading": null,
    "port_of_discharge": null,
    "destination": null,
    "lead_time": null,
    "estimated_delivery": null,
    "carrier": null,
    "tracking_number": null,
    "shipping_cost": null
  },
  "notes": null,
  "terms_and_conditions": null,
  "confidence": "High"
}

Rules:
- Use null for any field not present in the invoice.
- Numbers must be numeric (not strings): quantity, unit_price, subtotal, total, tax_amount, etc.
- For currency_conversions: if the invoice is NOT in USD, provide approximate USD and EUR equivalents using your knowledge of exchange rates. If already USD, set usd_rate to 1.0.
- For shipping: extract any logistics details such as shipping mode (air, sea, road, courier), Incoterms (FOB, CIF, EXW, etc.), ports, lead time, estimated delivery date, carrier, tracking number, and shipping cost. Use null if not present.
- notes: any free-text remarks, comments, or miscellaneous information on the document not captured elsewhere.
- terms_and_conditions: the full terms and conditions, payment terms text, or legal fine-print printed on the document. Also capture payment condition clauses (e.g. "Net 30", "50% advance", "due on delivery") here if they appear as paragraph text rather than a single field.
- confidence: High if most fields extracted cleanly, Medium if some ambiguity, Low if document is unclear."""



def _empty_invoice_result(filename: str, notes: str = "") -> dict:
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
        "shipping": {k: None for k in ["shipping_mode", "incoterms", "port_of_loading", "port_of_discharge", "destination", "lead_time", "estimated_delivery", "carrier", "tracking_number", "shipping_cost"]},
        "notes": notes or None,
        "terms_and_conditions": None,
        "input_tokens": 0, "output_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class InvoiceExtractor:
    def __init__(self):
        import os
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )

    @_timed("extract [total]")
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

    @_timed("_from_text")
    def _from_text(self, file_obj: io.BytesIO, filename: str, file_type: str) -> dict:
        file_obj.seek(0)
        raw_bytes = file_obj.read()

        if file_type == "pdf":
            return self._from_pdf_vision(io.BytesIO(raw_bytes), filename)

        from core.ingestor import DocumentIngestor
        text, _ = DocumentIngestor()._extract_fileobj(io.BytesIO(raw_bytes), f".{file_type}")
        response = self.client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract all invoice data from this document:\n\n{text[:30000]}"},
            ],
        )
        return self._parse(response, filename)

    @_timed("_from_pdf_vision")
    def _from_pdf_vision(self, file_obj: io.BytesIO, filename: str) -> dict:
        """Render each PDF page to JPEG via fitz and send all pages to the vision LLM."""
        file_obj.seek(0)
        raw_bytes = file_obj.read()

        try:
            import fitz
        except ImportError:
            return _empty_invoice_result(filename, "pymupdf is required for PDF extraction: pip install pymupdf")

        t0 = time.perf_counter()
        content_blocks = []
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= 4:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            b64 = base64.standard_b64encode(pix.tobytes("jpeg")).decode("utf-8")
            content_blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        doc.close()
        print(f"[invoice_extractor]   render+encode ({len(content_blocks)} pages): {time.perf_counter()-t0:.2f}s")

        if not content_blocks:
            return _empty_invoice_result(filename, "Could not extract text or render pages from this PDF.")

        content_blocks.append(
            {"type": "text", "text": EXTRACTION_PROMPT + "\n\nExtract all invoice data from these invoice page images."}
        )
        t4 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": content_blocks}],
        )
        print(f"[invoice_extractor]   api_call (vision, {len(content_blocks)-1} pages): {time.perf_counter()-t4:.2f}s")
        return self._parse(response, filename)

    @_timed("_from_image")
    def _from_image(self, file_obj: io.BytesIO, filename: str, file_type: str) -> dict:
        file_obj.seek(0)
        b64 = base64.standard_b64encode(file_obj.read()).decode("utf-8")
        mime = "image/jpeg" if file_type in {"jpg", "jpeg"} else f"image/{file_type}"

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

    @_timed("_parse")
    def _parse(self, response, filename: str) -> dict:
        raw = strip_json_fences(response.choices[0].message.content or "")
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"parse_error": "Could not parse AI response as JSON", "raw_snippet": raw[:400], "line_items": []}

        data["filename"] = filename
        data["input_tokens"] = input_tokens
        data["output_tokens"] = output_tokens
        return data
