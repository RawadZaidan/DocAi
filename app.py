import os
import io
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.ingestor import DocumentIngestor
from core.items_parser import ItemsParser
from core.query_engine import QueryEngine
from core.returnable_docs import ReturnableDocsExtractor
from core.legal_analyzer import LegalAnalyzer, LEGAL_LABEL_KEYWORDS
from core.invoice_extractor import InvoiceExtractor
from core.codemap import CodemapBuilder
from core.mindmap import MindMapBuilder

st.set_page_config(page_title="DocAI", layout="wide")

# ---------- Session state ----------
st.session_state.setdefault("active_mode", "🏗️ Tender")
# Tender
st.session_state.setdefault("docs", None)
st.session_state.setdefault("returnable_docs", None)
st.session_state.setdefault("financial_results", None)
st.session_state.setdefault("codemap", "")
st.session_state.setdefault("mindmap", "")
st.session_state.setdefault("qa_history", [])
# Legal
st.session_state.setdefault("legal_docs", None)
st.session_state.setdefault("legal_result", None)
st.session_state.setdefault("legal_qa_history", [])
# Invoice
st.session_state.setdefault("invoice_results", [])
# Shared
st.session_state.setdefault("errors", [])

# ---------- Colour maps ----------
TENDER_LABEL_COLORS = {
    "Technical Specifications": "#e3f2fd",
    "Financial Template": "#e8f5e9",
    "Legal / Eligibility Forms": "#fffde7",
    "Instructions to Bidders": "#f3e5f5",
    "Annexes / Returnables": "#fff3e0",
    "Unknown": "#f5f5f5",
}
LEGAL_LABEL_COLORS = {
    "Contract / Agreement": "#e3f2fd",
    "NDA / Confidentiality": "#fce4ec",
    "MOU / Letter of Intent": "#f3e5f5",
    "Policy / Regulation": "#fff3e0",
    "License Agreement": "#e8f5e9",
    "Service Agreement / SOW": "#e0f7fa",
    "Corporate / Legal Forms": "#fffde7",
    "Other Legal": "#f5f5f5",
}
RISK_COLORS = {"High": "#ffebee", "Medium": "#fff3e0", "Low": "#fffde7"}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    mode = st.radio(
        "Mode",
        ["🏗️ Tender", "⚖️ Legal", "🧾 Invoice"],
        index=["🏗️ Tender", "⚖️ Legal", "🧾 Invoice"].index(
            st.session_state["active_mode"]
        ),
        label_visibility="collapsed",
    )
    if mode != st.session_state["active_mode"]:
        st.session_state["active_mode"] = mode
        st.session_state["errors"] = []

    st.markdown("---")

    # ── Tender sidebar ─────────────────────────────────────
    if mode == "🏗️ Tender":
        st.markdown("### 🏗️ Tender Intelligence")
        tender_files = st.file_uploader(
            "📁 Upload Tender Documents",
            type=["pdf", "docx", "xlsx", "csv"],
            accept_multiple_files=True,
            help="PDF, DOCX, XLSX, CSV — full tender package.",
        )
        tender_clicked = st.button(
            "🔍 Analyze Tender",
            use_container_width=True,
            disabled=not tender_files,
        )
        if st.session_state["docs"] is not None:
            st.markdown("---")
            docs_ss = st.session_state["docs"]
            st.metric("Files Loaded", len(docs_ss))
            label_counts: dict = {}
            for meta in docs_ss.values():
                lbl = meta.get("label", "Unknown")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            for lbl, cnt in label_counts.items():
                st.metric(lbl, cnt)

    # ── Legal sidebar ───────────────────────────────────────
    elif mode == "⚖️ Legal":
        st.markdown("### ⚖️ Legal Document Analyzer")
        legal_files = st.file_uploader(
            "📁 Upload Legal Documents",
            type=["pdf", "docx", "xlsx", "csv"],
            accept_multiple_files=True,
            help="Upload contracts, NDAs, policies, agreements — any batch size.",
        )
        legal_clicked = st.button(
            "⚖️ Analyze Documents",
            use_container_width=True,
            disabled=not legal_files,
        )
        if st.session_state["legal_docs"] is not None:
            st.markdown("---")
            ld = st.session_state["legal_docs"]
            st.metric("Files Loaded", len(ld))
            lbl_counts: dict = {}
            for meta in ld.values():
                lbl = meta.get("label", "Other Legal")
                lbl_counts[lbl] = lbl_counts.get(lbl, 0) + 1
            for lbl, cnt in lbl_counts.items():
                st.metric(lbl, cnt)

    # ── Invoice sidebar ─────────────────────────────────────
    elif mode == "🧾 Invoice":
        st.markdown("### 🧾 Invoice Extractor")
        invoice_files = st.file_uploader(
            "📁 Upload Invoice(s)",
            type=["pdf", "docx", "xlsx", "csv", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="PDF, DOCX, Excel, or image (PNG/JPG/WEBP). Multiple invoices supported.",
        )
        invoice_clicked = st.button(
            "🧾 Extract Invoice Data",
            use_container_width=True,
            disabled=not invoice_files,
        )
        if st.session_state["invoice_results"]:
            st.markdown("---")
            st.metric("Invoices Processed", len(st.session_state["invoice_results"]))
            total_tokens_in = sum(
                r.get("input_tokens", 0) for r in st.session_state["invoice_results"]
            )
            st.metric("Total Tokens Used", f"{total_tokens_in:,}")

    # ── Shared error expander ───────────────────────────────
    if st.session_state["errors"]:
        st.markdown("---")
        with st.expander(f"⚠️ Processing Errors ({len(st.session_state['errors'])})"):
            for err in st.session_state["errors"]:
                st.error(f"**{err['file']}**: {err['error']}")


# ============================================================
# PIPELINES
# ============================================================

# ── Tender pipeline ─────────────────────────────────────────
if mode == "🏗️ Tender" and tender_clicked:
    st.session_state["errors"] = []
    with st.spinner("Loading documents..."):
        progress = st.progress(0)
        ingestor = DocumentIngestor()
        docs = ingestor.load_files(tender_files)
        progress.progress(20)

    if not docs:
        st.error("No supported documents found in the upload.")
    else:
        with st.spinner("Classifying documents..."):
            docs = ingestor.classify_documents(docs)
            progress.progress(40)

        with st.spinner("Identifying returnable documents..."):
            rde = ReturnableDocsExtractor()
            returnable_docs = rde.extract(docs)
            progress.progress(60)

        with st.spinner("Parsing items and lots..."):
            ip = ItemsParser()
            financial_results = ip.parse(docs)
            progress.progress(90)

        with st.spinner("Building codemap and mind map..."):
            codemap = CodemapBuilder().build(docs)
            mindmap = MindMapBuilder().build(docs)
            progress.progress(100)

        st.session_state["docs"] = docs
        st.session_state["returnable_docs"] = returnable_docs
        st.session_state["financial_results"] = financial_results
        st.session_state["codemap"] = codemap
        st.session_state["mindmap"] = mindmap
        st.success(f"✅ Analysis complete. {len(docs)} files processed.")

# ── Legal pipeline ───────────────────────────────────────────
if mode == "⚖️ Legal" and legal_clicked:
    st.session_state["errors"] = []
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.error("Add `OPENROUTER_API_KEY` to your `.env` file to enable Legal analysis.")
    else:
        with st.spinner("Loading documents..."):
            progress = st.progress(0)
            ingestor = DocumentIngestor()
            legal_docs = ingestor.load_files(legal_files)
            progress.progress(25)

        if not legal_docs:
            st.error("No supported documents found in the upload.")
        else:
            with st.spinner("Classifying documents..."):
                la = LegalAnalyzer()
                legal_docs = la.classify_documents(legal_docs)
                progress.progress(50)

            with st.spinner("Running AI legal analysis (this may take 30–60 s for large batches)..."):
                legal_result = la.analyze(legal_docs)
                progress.progress(100)

            st.session_state["legal_docs"] = legal_docs
            st.session_state["legal_result"] = legal_result
            st.session_state["legal_qa_history"] = []

            if legal_result.get("error"):
                st.warning(f"⚠️ Analysis error: {legal_result['error']}")
            else:
                st.success(
                    f"✅ Analysis complete. {len(legal_docs)} files · "
                    f"{legal_result.get('input_tokens', 0):,} tokens in / "
                    f"{legal_result.get('output_tokens', 0):,} out"
                )

# ── Invoice pipeline ─────────────────────────────────────────
if mode == "🧾 Invoice" and invoice_clicked:
    st.session_state["errors"] = []
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.error("Add `OPENROUTER_API_KEY` to your `.env` file to enable Invoice extraction.")
    else:
        extractor = InvoiceExtractor()
        results = []
        progress = st.progress(0)
        for i, uf in enumerate(invoice_files):
            fname = uf.name
            ext = os.path.splitext(fname)[1].lower().lstrip(".")
            with st.spinner(f"Extracting: {fname}..."):
                file_obj = io.BytesIO(uf.read())
                result = extractor.extract(file_obj, fname, ext)
                results.append(result)
                if result.get("error"):
                    st.session_state["errors"].append(
                        {"file": fname, "error": result["error"]}
                    )
            progress.progress(int((i + 1) / len(invoice_files) * 100))

        st.session_state["invoice_results"] = results
        ok = sum(1 for r in results if not r.get("error"))
        st.success(f"✅ Extracted {ok}/{len(results)} invoices.")


# ============================================================
# MAIN TABS — TENDER
# ============================================================
if mode == "🏗️ Tender":
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Documents", "📋 Returnable Docs", "📦 Items", "🤖 Q&A",
    ])

    # ── Tab 1: Document Overview ───────────────────────────
    with tab1:
        st.header("Document Overview")
        if st.session_state["docs"] is None:
            st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
        else:
            docs = st.session_state["docs"]
            rows = []
            for fname, meta in docs.items():
                rows.append({
                    "Filename": fname,
                    "Type": meta.get("label", "Unknown"),
                    "Pages/Sheets": len(meta.get("pages", [])),
                    "Confidence": f"{meta.get('confidence', 0):.1%}",
                    "Char Count": f"{meta.get('char_count', 0):,}",
                })
            df = pd.DataFrame(rows)

            def _tender_row_color(row):
                color = TENDER_LABEL_COLORS.get(row["Type"], "#f5f5f5")
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df.style.apply(_tender_row_color, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            cm = st.session_state.get("codemap", "")
            mm = st.session_state.get("mindmap", "")

            col_cm, col_mm = st.columns(2)
            with col_cm:
                with st.expander("🗺️ Document Codemap", expanded=False):
                    if cm:
                        st.code(cm, language=None)
                    else:
                        st.info("Codemap not yet generated.")
            with col_mm:
                with st.expander("🧠 Tender Mind Map", expanded=False):
                    if mm:
                        st.code(mm, language=None)
                    else:
                        st.info("Mind map not yet generated.")

    # ── Tab 2: Returnable Docs ─────────────────────────────
    with tab2:
        st.header("Returnable Documents")
        if st.session_state["returnable_docs"] is None:
            st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
        else:
            rd = st.session_state["returnable_docs"]
            rde = ReturnableDocsExtractor()
            items = rd.get("items", [])

            method = rd.get("method", "regex")
            if method == "claude":
                in_tok = rd.get("input_tokens", 0)
                out_tok = rd.get("output_tokens", 0)
                st.success(
                    f"🤖 **AI-powered extraction** — {in_tok:,} tokens in / {out_tok:,} out"
                )
            else:
                st.info("🔍 **Regex-based extraction** (add `OPENROUTER_API_KEY` for AI mode)")

            if rd.get("error"):
                st.warning(f"⚠️ {rd['error']}")

            if not items:
                st.info("No returnable documents detected.")
            else:
                missing = [i for i in items if not i.get("found_in_package")]
                found = [i for i in items if i.get("found_in_package")]
                all_cats = sorted({i.get("category", "Other") for i in items})

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Returnables", len(items))
                m2.metric("Mandatory", sum(1 for i in items if i.get("mandatory")))
                m3.metric("❌ Missing", len(missing))
                m4.metric("✅ Found", len(found))
                st.markdown("---")

                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    sel_cats = st.multiselect("Filter by Category", options=all_cats, default=all_cats)
                with fcol2:
                    show_status = st.selectbox(
                        "Filter by Status", ["All", "❌ Missing only", "✅ Found only"]
                    )

                filtered = [
                    i for i in items
                    if i.get("category", "Other") in sel_cats
                    and (
                        show_status == "All"
                        or (show_status == "❌ Missing only" and not i.get("found_in_package"))
                        or (show_status == "✅ Found only" and i.get("found_in_package"))
                    )
                ]

                grouped: dict = {}
                for item in filtered:
                    grouped.setdefault(item.get("category", "Other"), []).append(item)

                CATEGORY_ICONS = {
                    "Certifications": "🏅", "Forms": "📝",
                    "Financial Documents": "💰", "Legal / Corporate": "⚖️",
                    "Technical Documents": "🔧", "Experience / References": "📂",
                    "Other": "📌",
                }

                for cat, cat_items in sorted(grouped.items()):
                    icon = CATEGORY_ICONS.get(cat, "📌")
                    missing_cnt = sum(1 for i in cat_items if not i.get("found_in_package"))
                    label = f"{icon} {cat}  —  {len(cat_items)} item(s)"
                    if missing_cnt:
                        label += f"  ·  **{missing_cnt} missing**"
                    with st.expander(label, expanded=(missing_cnt > 0)):
                        for item in cat_items:
                            status_icon = "✅" if item.get("found_in_package") else "❌"
                            mandatory_badge = "🔴 Mandatory" if item.get("mandatory") else "🟡 Optional"
                            st.markdown(f"{status_icon} **{item['doc_name']}** &nbsp;&nbsp; {mandatory_badge}")
                            if item.get("description"):
                                st.caption(item["description"])
                            st.caption(f"Mentioned in: {item.get('source_file', '—')}")
                            st.markdown("")

                st.markdown("---")
                with st.expander("📊 Full Table View"):
                    full_df = rde.to_dataframe(rd)
                    view_df = full_df.copy()
                    if sel_cats:
                        view_df = view_df[view_df["Category"].isin(sel_cats)]
                    if show_status == "❌ Missing only":
                        view_df = view_df[view_df["Status"] == "❌ Missing"]
                    elif show_status == "✅ Found only":
                        view_df = view_df[view_df["Status"] == "✅ Found"]

                    def _rd_color(row):
                        if row["Status"] == "❌ Missing":
                            return ["background-color: #ffebee"] * len(row)
                        return ["background-color: #e8f5e9"] * len(row)

                    st.dataframe(view_df.style.apply(_rd_color, axis=1), use_container_width=True, hide_index=True)

                full_df2 = rde.to_dataframe(rd)
                st.download_button(
                    "⬇ Export Returnables CSV",
                    data=full_df2.to_csv(index=False),
                    file_name="returnable_docs.csv",
                    mime="text/csv",
                )

            with st.expander("📂 Documents analyzed"):
                for fn in rd.get("docs_included", []):
                    st.write(f"  • {fn}")
                skp = rd.get("docs_skipped", [])
                if skp:
                    st.write("**Skipped (token budget):**")
                    for fn in skp:
                        st.write(f"  • {fn}")

    # ── Tab 3: Items ───────────────────────────────────────
    with tab3:
        st.header("Items by Lot")
        if st.session_state["financial_results"] is None:
            st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
        else:
            lots = st.session_state["financial_results"]
            if not lots:
                st.info("No item tables detected in this tender package.")
            else:
                files_seen: dict = {}
                for lot in lots:
                    files_seen.setdefault(lot["filename"], []).append(lot)

                for fname, file_lots in files_seen.items():
                    total_items = sum(len(l["items"]) for l in file_lots)
                    st.subheader(f"{fname}  —  {len(file_lots)} lot(s) · {total_items} item(s)")

                    for lot in file_lots:
                        df = lot.get("items", pd.DataFrame())
                        error = lot.get("error")
                        with st.expander(f"📦 {lot['lot_name']}  ({len(df)} items)", expanded=True):
                            if error:
                                st.error(f"🔴 Parse error: {error}")
                            elif df.empty:
                                st.info("No items extracted for this lot.")
                            else:
                                keep = [c for c in ["Description", "Quantity", "Unit Price"] if c in df.columns]
                                display_df = df[keep].reset_index(drop=True)
                                display_df.insert(0, "#", range(1, len(display_df) + 1))
                                st.dataframe(display_df, use_container_width=True, hide_index=True)
                                safe_lot = re.sub(r"[^\w\-]", "_", lot["lot_name"])
                                st.download_button(
                                    f"⬇ Export {lot['lot_name']}",
                                    data=display_df.to_csv(index=False),
                                    file_name=f"{safe_lot}.csv",
                                    mime="text/csv",
                                    key=f"dl_{fname}_{safe_lot}",
                                )
                    st.markdown("---")

    # ── Tab 4: Q&A ─────────────────────────────────────────
    with tab4:
        st.header("Q&A Engine")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            st.info("Add `OPENROUTER_API_KEY` to `.env` to enable Q&A.")
            st.stop()

        if st.session_state["docs"] is None:
            st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
        else:
            with st.form("qa_form", clear_on_submit=False):
                question = st.text_area("Ask a question about this tender...", height=80)
                ask_clicked = st.form_submit_button("Ask")

            if ask_clicked and question.strip():
                with st.spinner("Analyzing tender documents..."):
                    qe = QueryEngine()
                    try:
                        result = qe.answer(
                            question.strip(),
                            st.session_state["docs"],
                            codemap=st.session_state.get("codemap", ""),
                            mindmap=st.session_state.get("mindmap", ""),
                        )
                        st.session_state["qa_history"].append({
                            "question": question.strip(),
                            "answer": result["answer"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "docs_included": result["docs_included"],
                            "docs_skipped": result["docs_skipped"],
                            "corpus_coverage": result.get("corpus_coverage", 0.0),
                        })
                    except Exception as e:
                        st.error(f"API error: {e}")

            if st.session_state["qa_history"]:
                latest = st.session_state["qa_history"][-1]
                st.markdown(latest["answer"])
                coverage = latest.get("corpus_coverage", 0.0)
                st.caption(
                    f"Tokens used: {latest['input_tokens']:,} in / {latest['output_tokens']:,} out"
                    f"  ·  Corpus coverage: {coverage:.0%} of documents included in full"
                )
                with st.expander("📂 Documents included in this query"):
                    for fn in latest.get("docs_included", []):
                        st.write(f"  • {fn}")
                    if latest.get("docs_skipped"):
                        st.write("**Skipped (token budget):**")
                        for fn in latest["docs_skipped"]:
                            st.write(f"  • {fn}")

                if len(st.session_state["qa_history"]) > 1:
                    st.markdown("---")
                    st.subheader("Previous Questions")
                    for i, entry in enumerate(reversed(st.session_state["qa_history"][:-1]), 1):
                        with st.expander(f"Q{i}: {entry['question'][:80]}"):
                            st.markdown(entry["answer"])


# ============================================================
# MAIN TABS — LEGAL
# ============================================================
elif mode == "⚖️ Legal":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Documents", "👥 Parties", "📅 Key Dates",
        "⚠️ Risk Analysis", "📋 Obligations", "🤖 Q&A",
    ])

    legal_docs = st.session_state.get("legal_docs")
    legal_result = st.session_state.get("legal_result")

    _no_data_msg = "Run **⚖️ Analyze Documents** from the sidebar to begin."

    # ── Tab 1: Document Overview ───────────────────────────
    with tab1:
        st.header("Document Overview")
        if legal_docs is None:
            st.info(_no_data_msg)
        else:
            rows = []
            for fname, meta in legal_docs.items():
                rows.append({
                    "Filename": fname,
                    "Type": meta.get("label", "Other Legal"),
                    "Pages/Sheets": len(meta.get("pages", [])),
                    "Confidence": f"{meta.get('confidence', 0):.1%}",
                    "Char Count": f"{meta.get('char_count', 0):,}",
                })
            df = pd.DataFrame(rows)

            def _legal_row_color(row):
                color = LEGAL_LABEL_COLORS.get(row["Type"], "#f5f5f5")
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df.style.apply(_legal_row_color, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            if legal_result and legal_result.get("summary"):
                st.markdown("---")
                st.subheader("AI Summary")
                st.info(legal_result["summary"])

    # ── Tab 2: Parties ─────────────────────────────────────
    with tab2:
        st.header("Parties & Roles")
        if legal_result is None:
            st.info(_no_data_msg)
        else:
            parties = legal_result.get("parties", [])
            if not parties:
                st.info("No parties identified in these documents.")
            else:
                st.metric("Parties Identified", len(parties))
                for p in parties:
                    with st.expander(f"👤 {p.get('name', '—')}  —  *{p.get('role', '')}*", expanded=True):
                        st.write(p.get("description", ""))

    # ── Tab 3: Key Dates ────────────────────────────────────
    with tab3:
        st.header("Key Dates & Deadlines")
        if legal_result is None:
            st.info(_no_data_msg)
        else:
            dates = legal_result.get("key_dates", [])
            if not dates:
                st.info("No key dates identified in these documents.")
            else:
                df_dates = pd.DataFrame(dates)[
                    [c for c in ["event", "date", "source"] if c in pd.DataFrame(dates).columns]
                ]
                df_dates.columns = [c.title() for c in df_dates.columns]
                st.dataframe(df_dates, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Export Dates CSV",
                    data=df_dates.to_csv(index=False),
                    file_name="key_dates.csv",
                    mime="text/csv",
                )

    # ── Tab 4: Risk Analysis ────────────────────────────────
    with tab4:
        st.header("Risk Analysis")
        if legal_result is None:
            st.info(_no_data_msg)
        else:
            risks = legal_result.get("risk_clauses", [])
            if not risks:
                st.success("✅ No significant risk clauses identified.")
            else:
                high = [r for r in risks if r.get("risk_level") == "High"]
                med = [r for r in risks if r.get("risk_level") == "Medium"]
                low = [r for r in risks if r.get("risk_level") == "Low"]

                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 High Risk", len(high))
                c2.metric("🟡 Medium Risk", len(med))
                c3.metric("🟢 Low Risk", len(low))
                st.markdown("---")

                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    ["High", "Medium", "Low"],
                    default=["High", "Medium", "Low"],
                )
                filtered_risks = [r for r in risks if r.get("risk_level") in risk_filter]

                for r in filtered_risks:
                    level = r.get("risk_level", "Low")
                    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")
                    with st.expander(
                        f"{color} **{r.get('type', 'Unknown')}** — {level} Risk  |  {r.get('source', '')}",
                        expanded=(level == "High"),
                    ):
                        st.markdown(f"**Clause excerpt:**")
                        st.code(r.get("clause_excerpt", ""), language=None)
                        st.markdown(f"**Why it matters:** {r.get('explanation', '')}")

                st.markdown("---")
                risk_df = pd.DataFrame(filtered_risks)
                if not risk_df.empty:
                    st.download_button(
                        "⬇ Export Risk Report CSV",
                        data=risk_df.to_csv(index=False),
                        file_name="risk_report.csv",
                        mime="text/csv",
                    )

    # ── Tab 5: Obligations ──────────────────────────────────
    with tab5:
        st.header("Obligations")
        if legal_result is None:
            st.info(_no_data_msg)
        else:
            obligations = legal_result.get("obligations", [])
            if not obligations:
                st.info("No obligations extracted from these documents.")
            else:
                all_parties = sorted({o.get("party", "Unknown") for o in obligations})
                sel_party = st.multiselect(
                    "Filter by Party", options=all_parties, default=all_parties
                )
                filtered_obs = [o for o in obligations if o.get("party") in sel_party]

                mandatory_count = sum(1 for o in filtered_obs if o.get("mandatory"))
                st.metric("Total Obligations", len(filtered_obs))
                st.metric("Mandatory", mandatory_count)

                df_obs = pd.DataFrame(filtered_obs)
                if not df_obs.empty:
                    col_order = [c for c in ["party", "obligation", "mandatory", "source"] if c in df_obs.columns]
                    df_obs = df_obs[col_order]
                    df_obs.columns = [c.title() for c in df_obs.columns]

                    def _obs_color(row):
                        if "Mandatory" in row.index and row["Mandatory"]:
                            return ["background-color: #ffebee"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        df_obs.style.apply(_obs_color, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "⬇ Export Obligations CSV",
                        data=df_obs.to_csv(index=False),
                        file_name="obligations.csv",
                        mime="text/csv",
                    )

            defined_terms = legal_result.get("defined_terms", [])
            if defined_terms:
                st.markdown("---")
                st.subheader("Defined Terms")
                df_terms = pd.DataFrame(defined_terms)
                if not df_terms.empty:
                    df_terms.columns = [c.title() for c in df_terms.columns]
                    st.dataframe(df_terms, use_container_width=True, hide_index=True)

    # ── Tab 6: Q&A ─────────────────────────────────────────
    with tab6:
        st.header("Legal Q&A")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            st.info("Add `OPENROUTER_API_KEY` to `.env` to enable Q&A.")
        elif legal_docs is None:
            st.info(_no_data_msg)
        else:
            with st.form("legal_qa_form", clear_on_submit=False):
                lq = st.text_area(
                    "Ask a question about these legal documents...", height=80,
                    placeholder="e.g. What are the termination conditions? Who bears liability for delays?"
                )
                lask = st.form_submit_button("Ask")

            if lask and lq.strip():
                with st.spinner("Analyzing documents..."):
                    la2 = LegalAnalyzer()
                    try:
                        result = la2.answer(lq.strip(), legal_docs)
                        st.session_state["legal_qa_history"].append({
                            "question": lq.strip(),
                            "answer": result["answer"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                        })
                    except Exception as e:
                        st.error(f"API error: {e}")

            if st.session_state["legal_qa_history"]:
                latest = st.session_state["legal_qa_history"][-1]
                st.markdown(latest["answer"])
                st.caption(
                    f"Tokens: {latest['input_tokens']:,} in / {latest['output_tokens']:,} out"
                )
                if len(st.session_state["legal_qa_history"]) > 1:
                    st.markdown("---")
                    st.subheader("Previous Questions")
                    for i, entry in enumerate(reversed(st.session_state["legal_qa_history"][:-1]), 1):
                        with st.expander(f"Q{i}: {entry['question'][:80]}"):
                            st.markdown(entry["answer"])


# ============================================================
# MAIN TABS — INVOICE
# ============================================================
elif mode == "🧾 Invoice":
    invoice_results = st.session_state.get("invoice_results", [])

    if not invoice_results:
        st.info(
            "Upload invoice files in the sidebar and click **🧾 Extract Invoice Data** to begin.\n\n"
            "Supported: PDF, DOCX, Excel, CSV, and image files (PNG, JPG, WEBP)."
        )
    else:
        # Invoice selector when multiple files
        if len(invoice_results) > 1:
            inv_names = [r.get("filename", f"Invoice {i+1}") for i, r in enumerate(invoice_results)]
            selected_name = st.selectbox("Select Invoice", inv_names)
            inv = next(r for r in invoice_results if r.get("filename") == selected_name)
        else:
            inv = invoice_results[0]

        if inv.get("error"):
            st.error(f"Extraction failed: {inv['error']}")
        elif inv.get("parse_error"):
            st.warning(f"⚠️ Could not parse structured response. Raw output below:")
            st.code(inv.get("raw_snippet", ""))
        else:
            fname_display = inv.get("filename", "Invoice")
            st.subheader(f"🧾 {fname_display}")
            conf = inv.get("confidence", "—")
            conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
            st.caption(
                f"{conf_color} Extraction confidence: **{conf}** · "
                f"{inv.get('input_tokens', 0):,} tokens used"
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "🧾 Summary", "📊 Line Items", "💱 Currency", "🏦 Bank & Notes"
            ])

            # ── Invoice Tab 1: Summary ─────────────────────
            with tab1:
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("##### Invoice Details")
                    fields = [
                        ("Invoice #", "invoice_number"),
                        ("Invoice Date", "invoice_date"),
                        ("Due Date", "due_date"),
                        ("Payment Terms", "payment_terms"),
                        ("Validity", "validity"),
                        ("PO Number", "po_number"),
                    ]
                    for label, key in fields:
                        val = inv.get(key)
                        if val:
                            st.markdown(f"**{label}:** {val}")

                    st.markdown("---")
                    st.markdown("##### Issued By")
                    issuer = inv.get("issuer") or {}
                    for label, key in [
                        ("Name", "name"), ("Address", "address"), ("City", "city"),
                        ("Country", "country"), ("Phone", "phone"), ("Email", "email"),
                        ("Website", "website"), ("Tax ID", "tax_id"), ("Registration", "registration"),
                    ]:
                        val = issuer.get(key)
                        if val:
                            st.markdown(f"**{label}:** {val}")

                with c2:
                    st.markdown("##### Billed To")
                    client = inv.get("client") or {}
                    for label, key in [
                        ("Name", "name"), ("Contact", "contact_person"),
                        ("Address", "address"), ("City", "city"), ("Country", "country"),
                        ("Phone", "phone"), ("Email", "email"), ("Tax ID", "tax_id"),
                    ]:
                        val = client.get(key)
                        if val:
                            st.markdown(f"**{label}:** {val}")

                    st.markdown("---")
                    st.markdown("##### Totals")
                    currency_sym = inv.get("currency_symbol") or inv.get("currency") or ""
                    for label, key in [
                        ("Subtotal", "subtotal"), ("Discount", "discount"),
                        ("Tax", "tax_amount"), ("**Total**", "total"),
                    ]:
                        val = inv.get(key)
                        if val is not None:
                            st.markdown(f"**{label}:** {currency_sym} {val:,.2f}" if isinstance(val, (int, float)) else f"**{label}:** {val}")

            # ── Invoice Tab 2: Line Items ──────────────────
            with tab2:
                st.subheader("Line Items")
                line_items = inv.get("line_items") or []
                if not line_items:
                    st.info("No line items extracted.")
                else:
                    df_items = pd.DataFrame(line_items)
                    # Rename columns for display
                    rename_map = {
                        "description": "Description", "quantity": "Qty",
                        "unit": "Unit", "unit_price": "Unit Price",
                        "subtotal": "Subtotal", "tax_rate": "Tax Rate",
                    }
                    df_items = df_items.rename(columns={k: v for k, v in rename_map.items() if k in df_items.columns})
                    df_items.insert(0, "#", range(1, len(df_items) + 1))
                    st.dataframe(df_items, use_container_width=True, hide_index=True)

                    st.download_button(
                        "⬇ Export Line Items CSV",
                        data=df_items.to_csv(index=False),
                        file_name=f"{os.path.splitext(fname_display)[0]}_line_items.csv",
                        mime="text/csv",
                    )

            # ── Invoice Tab 3: Currency ────────────────────
            with tab3:
                st.subheader("Currency & Conversions")
                currency = inv.get("currency") or "—"
                currency_sym = inv.get("currency_symbol") or ""
                total = inv.get("total")

                st.markdown(f"**Invoice Currency:** {currency_sym} {currency}")
                if total is not None and isinstance(total, (int, float)):
                    st.metric("Total Amount", f"{currency_sym} {total:,.2f}")

                conv = inv.get("currency_conversions") or {}
                if conv.get("total_usd") or conv.get("total_eur"):
                    st.markdown("---")
                    st.markdown("##### Approximate Conversions")
                    col1, col2 = st.columns(2)
                    if conv.get("total_usd") is not None:
                        rate = conv.get("usd_rate", "")
                        col1.metric("USD", f"$ {conv['total_usd']:,.2f}", delta=f"Rate: {rate}" if rate else None)
                    if conv.get("total_eur") is not None:
                        rate = conv.get("eur_rate", "")
                        col2.metric("EUR", f"€ {conv['total_eur']:,.2f}", delta=f"Rate: {rate}" if rate else None)
                    note = conv.get("rate_note", "")
                    if note:
                        st.caption(f"ℹ️ {note}")
                else:
                    st.info("Currency conversion not available for this invoice.")

            # ── Invoice Tab 4: Bank & Notes ────────────────
            with tab4:
                bank = inv.get("bank_details") or {}
                if any(bank.values()):
                    st.subheader("Bank Details")
                    for label, key in [
                        ("Bank Name", "bank_name"), ("Account #", "account_number"),
                        ("IBAN", "iban"), ("SWIFT/BIC", "swift"), ("Routing #", "routing"),
                    ]:
                        val = bank.get(key)
                        if val:
                            st.markdown(f"**{label}:** `{val}`")
                else:
                    st.info("No bank details found in this invoice.")

                notes = inv.get("notes")
                if notes:
                    st.markdown("---")
                    st.subheader("Notes")
                    st.write(notes)

        # ── Summary table for multiple invoices ───────────
        if len(invoice_results) > 1:
            st.markdown("---")
            st.subheader("All Invoices — Summary")
            summary_rows = []
            for r in invoice_results:
                total = r.get("total")
                currency_sym = r.get("currency_symbol") or r.get("currency") or ""
                total_str = f"{currency_sym} {total:,.2f}" if isinstance(total, (int, float)) else str(total or "—")
                issuer_name = (r.get("issuer") or {}).get("name") or "—"
                client_name = (r.get("client") or {}).get("name") or "—"
                summary_rows.append({
                    "File": r.get("filename", "—"),
                    "Invoice #": r.get("invoice_number") or "—",
                    "Date": r.get("invoice_date") or "—",
                    "Issued By": issuer_name,
                    "Billed To": client_name,
                    "Total": total_str,
                    "Confidence": r.get("confidence") or "—",
                    "Status": "⚠️ Error" if r.get("error") else "✅ OK",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
