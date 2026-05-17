import re
import pandas as pd
from rapidfuzz import process as rfprocess


CANONICAL_CANDIDATES = {
    "description": ["description", "item", "scope", "work item", "particulars", "activity"],
    "unit": ["unit", "uom", "unit of measure", "measure"],
    "quantity": ["qty", "quantity", "no.", "nos", "number"],
    "unit_price": ["unit price", "rate", "price", "cost per unit", "unit cost", "unit rate"],
    "total": ["total", "amount", "total price", "subtotal", "extended price", "line total"],
}

CURRENCY_PATTERNS = re.compile(r'(\$|€|£|USD|EUR|GBP|LBP)', re.IGNORECASE)


class FinancialParser:
    def parse(self, docs: dict) -> list:
        results = []
        for fname, meta in docs.items():
            is_financial = (
                meta.get("label") == "Financial Template"
                or meta.get("file_type") in ("xlsx", "csv")
            )
            if not is_financial:
                continue
            try:
                df = self._to_dataframe(meta)
                if df is None or df.empty:
                    continue
                col_map = self.detect_columns(df)
                flags = self.validate(df, col_map)
                results.append({
                    "filename": fname,
                    "dataframe": df,
                    "column_map": col_map,
                    "flags": flags,
                })
            except Exception as e:
                results.append({
                    "filename": fname,
                    "dataframe": pd.DataFrame(),
                    "column_map": {},
                    "flags": [{"row": -1, "issue": "Parse Error", "detail": str(e)}],
                })
        return results

    def _to_dataframe(self, meta: dict) -> pd.DataFrame:
        file_type = meta.get("file_type")
        text = meta.get("text", "")
        if not text.strip():
            return None

        # For xlsx/csv, the text is already in markdown table form.
        # Try to parse the first markdown table found.
        lines = text.splitlines()
        table_lines = []
        in_table = False
        for line in lines:
            if re.match(r'^\s*\|', line):
                in_table = True
                table_lines.append(line)
            elif in_table:
                break

        if len(table_lines) >= 2:
            try:
                raw_header = [c.strip() for c in table_lines[0].strip('|').split('|')]
                # Deduplicate column names: blank or repeated cols get a numeric suffix
                seen = {}
                header = []
                for col in raw_header:
                    key = col if col else "_col"
                    count = seen.get(key, 0)
                    seen[key] = count + 1
                    header.append(col if count == 0 else f"{key}_{count}")
                data_rows = []
                for tl in table_lines[2:]:  # skip separator line
                    cells = [c.strip() for c in tl.strip('|').split('|')]
                    if len(cells) == len(header):
                        data_rows.append(cells)
                if data_rows:
                    return pd.DataFrame(data_rows, columns=header)
            except Exception:
                pass

        return None

    def detect_columns(self, df: pd.DataFrame) -> dict:
        col_map = {}
        actual_cols = list(df.columns)
        for canonical, candidates in CANONICAL_CANDIDATES.items():
            for actual in actual_cols:
                result = rfprocess.extractOne(
                    actual.lower(), [c.lower() for c in candidates]
                )
                if result and result[1] >= 70:
                    col_map[canonical] = actual
                    break
        return col_map

    def validate(self, df: pd.DataFrame, col_map: dict) -> list:
        flags = []

        # Math check
        if all(k in col_map for k in ("quantity", "unit_price", "total")):
            qty_col = col_map["quantity"]
            up_col = col_map["unit_price"]
            tot_col = col_map["total"]
            for idx, row in df.iterrows():
                try:
                    qty = float(str(row[qty_col]).replace(",", ""))
                    up = float(str(row[up_col]).replace(",", ""))
                    tot = float(str(row[tot_col]).replace(",", ""))
                    expected = qty * up
                    if tot != 0 and abs(expected - tot) / abs(tot) > 0.01:
                        flags.append({
                            "row": idx,
                            "issue": "Math Error",
                            "detail": (
                                f"Row {idx}: {qty} × {up} = {expected:.2f}, "
                                f"found {tot:.2f}"
                            ),
                        })
                except (ValueError, ZeroDivisionError):
                    continue

        # Currency inconsistency — scan all string cells and column headers
        all_text = " ".join(
            str(v) for v in df.values.flatten()
        ) + " " + " ".join(df.columns)
        all_text += " " + " ".join(df.columns)
        found = set(m.upper() for m in CURRENCY_PATTERNS.findall(all_text))
        # Normalise $ → USD-like
        currency_groups = {
            "USD_GROUP": {"$", "USD"},
            "EUR_GROUP": {"€", "EUR"},
            "GBP_GROUP": {"£", "GBP"},
            "LBP_GROUP": {"LBP"},
        }
        active_groups = [g for g, members in currency_groups.items() if found & members]
        if len(active_groups) > 1:
            flags.append({
                "row": -1,
                "issue": "Currency Inconsistency",
                "detail": f"Multiple currencies detected: {', '.join(sorted(found))}",
            })

        # Empty totals
        if "total" in col_map and "quantity" in col_map and "unit_price" in col_map:
            tot_col = col_map["total"]
            qty_col = col_map["quantity"]
            up_col = col_map["unit_price"]
            for idx, row in df.iterrows():
                tot_val = str(row[tot_col]).strip()
                try:
                    qty = float(str(row[qty_col]).replace(",", ""))
                    up = float(str(row[up_col]).replace(",", ""))
                except (ValueError, TypeError):
                    continue
                if (
                    (tot_val in ("", "nan", "None", "N/A"))
                    and qty != 0
                    and up != 0
                ):
                    flags.append({
                        "row": idx,
                        "issue": "Missing Total",
                        "detail": "Quantity and unit price present but total is empty",
                    })

        return flags
