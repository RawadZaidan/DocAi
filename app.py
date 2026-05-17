import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.ingestor import DocumentIngestor
from core.items_parser import ItemsParser
from core.query_engine import QueryEngine
from core.returnable_docs import ReturnableDocsExtractor

st.set_page_config(page_title="Tender Intelligence System", layout="wide")

# ---------- Session state defaults ----------
st.session_state.setdefault("docs", None)
st.session_state.setdefault("returnable_docs", None)
st.session_state.setdefault("financial_results", None)
st.session_state.setdefault("errors", [])
st.session_state.setdefault("qa_history", [])

# ---------- Label colour map ----------
LABEL_COLORS = {
    "Technical Specifications": "#e3f2fd",
    "Financial Template": "#e8f5e9",
    "Legal / Eligibility Forms": "#fffde7",
    "Instructions to Bidders": "#f3e5f5",
    "Annexes / Returnables": "#fff3e0",
    "Unknown": "#f5f5f5",
}

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Tender Intelligence System")
    st.markdown("---")
    folder_path = st.text_input(
        "📁 Tender Folder Path",
        placeholder="/path/to/tender/folder",
        help="Paste the absolute path to the folder containing your tender documents.",
    )
    analyze_clicked = st.button("🔍 Analyze Tender", use_container_width=True)

    if st.session_state["docs"] is not None:
        st.markdown("---")
        docs = st.session_state["docs"]
        st.metric("Files Loaded", len(docs))
        label_counts = {}
        for meta in docs.values():
            lbl = meta.get("label", "Unknown")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        for lbl, cnt in label_counts.items():
            st.metric(lbl, cnt)

    if st.session_state["errors"]:
        with st.expander(f"⚠️ Processing Errors ({len(st.session_state['errors'])})"):
            for err in st.session_state["errors"]:
                st.error(f"**{err['file']}**: {err['error']}")

# ---------- Analysis pipeline ----------
if analyze_clicked:
    if not folder_path or not folder_path.strip():
        st.sidebar.error("Please enter a folder path.")
    else:
        st.session_state["errors"] = []

        with st.spinner("Loading documents..."):
            progress = st.progress(0)
            ingestor = DocumentIngestor()
            docs = ingestor.load_folder(folder_path.strip())
            progress.progress(20)

        if not docs:
            st.error("No supported documents found in the specified folder.")
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
                progress.progress(100)

            st.session_state["docs"] = docs
            st.session_state["returnable_docs"] = returnable_docs
            st.session_state["financial_results"] = financial_results

            st.success(f"✅ Analysis complete. {len(docs)} files processed.")

# ---------- Main tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Documents",
    "📋 Returnable Docs",
    "📦 Items",
    "🤖 Q&A",
])

# ===== TAB 1: Document Overview =====
with tab1:
    st.header("Document Overview")
    if st.session_state["docs"] is None:
        st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
    else:
        docs = st.session_state["docs"]
        rows = []
        for fname, meta in docs.items():
            page_count = len(meta.get("pages", []))
            rows.append({
                "Filename": fname,
                "Type": meta.get("label", "Unknown"),
                "Pages/Sheets": page_count,
                "Confidence": f"{meta.get('confidence', 0):.1%}",
                "Char Count": f"{meta.get('char_count', 0):,}",
            })
        df = pd.DataFrame(rows)

        def _row_color(row):
            color = LABEL_COLORS.get(row["Type"], "#f5f5f5")
            return [f"background-color: {color}"] * len(row)

        styled = df.style.apply(_row_color, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

# ===== TAB 2: Returnable Docs =====
with tab2:
    st.header("Returnable Documents")
    if st.session_state["returnable_docs"] is None:
        st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
    else:
        rd = st.session_state["returnable_docs"]
        rde = ReturnableDocsExtractor()
        items = rd.get("items", [])

        # Method banner
        method = rd.get("method", "regex")
        if method == "claude":
            in_tok = rd.get("input_tokens", 0)
            out_tok = rd.get("output_tokens", 0)
            st.success(
                f"🤖 **AI-powered extraction** via Claude — "
                f"{in_tok:,} tokens in / {out_tok:,} out"
            )
        else:
            st.info(
                "🔍 **Regex-based extraction** (add `OPENROUTER_API_KEY` to `.env` "
                "for AI-powered identification)"
            )

        if rd.get("error"):
            st.warning(f"⚠️ {rd['error']}")

        if not items:
            st.info("No returnable documents detected in this tender package.")
        else:
            missing = [i for i in items if not i.get("found_in_package")]
            found   = [i for i in items if i.get("found_in_package")]

            # Summary metrics
            all_cats = sorted({i.get("category", "Other") for i in items})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Returnables", len(items))
            m2.metric("Mandatory", sum(1 for i in items if i.get("mandatory")))
            m3.metric("❌ Missing", len(missing))
            m4.metric("✅ Found in Package", len(found))

            st.markdown("---")

            # Filter controls
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                sel_cats = st.multiselect(
                    "Filter by Category", options=all_cats, default=all_cats
                )
            with fcol2:
                show_status = st.selectbox(
                    "Filter by Status",
                    options=["All", "❌ Missing only", "✅ Found only"],
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

            # Category-grouped view
            grouped: dict[str, list] = {}
            for item in filtered:
                cat = item.get("category", "Other")
                grouped.setdefault(cat, []).append(item)

            CATEGORY_ICONS = {
                "Certifications": "🏅",
                "Forms": "📝",
                "Financial Documents": "💰",
                "Legal / Corporate": "⚖️",
                "Technical Documents": "🔧",
                "Experience / References": "📂",
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
                        st.markdown(
                            f"{status_icon} **{item['doc_name']}** &nbsp;&nbsp; {mandatory_badge}"
                        )
                        if item.get("description"):
                            st.caption(item["description"])
                        st.caption(f"Mentioned in: {item.get('source_file', '—')}")
                        st.markdown("")

            st.markdown("---")

            # Full table view
            with st.expander("📊 Full Table View"):
                df = rde.to_dataframe(rd)
                if sel_cats:
                    df = df[df["Category"].isin(sel_cats)]
                if show_status == "❌ Missing only":
                    df = df[df["Status"] == "❌ Missing"]
                elif show_status == "✅ Found only":
                    df = df[df["Status"] == "✅ Found"]

                def _rd_color(row):
                    if row["Status"] == "❌ Missing":
                        return ["background-color: #ffebee"] * len(row)
                    return ["background-color: #e8f5e9"] * len(row)

                styled = df.style.apply(_rd_color, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)

            full_df = rde.to_dataframe(rd)
            st.download_button(
                "⬇ Export Returnables CSV",
                data=full_df.to_csv(index=False),
                file_name="returnable_docs.csv",
                mime="text/csv",
            )

        # Docs context info
        with st.expander("📂 Documents analyzed for this extraction"):
            inc = rd.get("docs_included", [])
            skp = rd.get("docs_skipped", [])
            if inc:
                st.write("**Included:**")
                for fn in inc:
                    st.write(f"  • {fn}")
            if skp:
                st.write("**Skipped (token budget):**")
                for fn in skp:
                    st.write(f"  • {fn}")

# ===== TAB 3: Items =====
with tab3:
    st.header("Items by Lot")
    if st.session_state["financial_results"] is None:
        st.info("Run **🔍 Analyze Tender** from the sidebar to begin.")
    else:
        lots = st.session_state["financial_results"]
        if not lots:
            st.info("No item tables detected in this tender package.")
        else:
            # Group by filename for collapsible file sections
            files_seen = {}
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
                            keep = [c for c in ["Description", "Quantity", "Unit Price"]
                                    if c in df.columns]
                            display_df = df[keep].reset_index(drop=True)
                            display_df.insert(0, "#", range(1, len(display_df) + 1))
                            st.dataframe(display_df, use_container_width=True, hide_index=True)

                            csv_data = display_df.to_csv(index=False)
                            safe_lot = re.sub(r'[^\w\-]', '_', lot['lot_name'])
                            st.download_button(
                                f"⬇ Export {lot['lot_name']}",
                                data=csv_data,
                                file_name=f"{safe_lot}.csv",
                                mime="text/csv",
                                key=f"dl_{fname}_{safe_lot}",
                            )
                st.markdown("---")

# ===== TAB 4: Q&A Engine =====
with tab4:
    st.header("Q&A Engine")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.info(
            "Add `OPENROUTER_API_KEY` to your `.env` file (or environment) to enable Q&A. "
            "All other tabs work without it."
        )
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
                    result = qe.answer(question.strip(), st.session_state["docs"])
                    st.session_state["qa_history"].append(
                        {"question": question.strip(), "answer": result["answer"],
                         "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"],
                         "docs_included": result["docs_included"],
                         "docs_skipped": result["docs_skipped"]}
                    )
                except Exception as e:
                    st.error(f"API error: {e}")

        if st.session_state["qa_history"]:
            latest = st.session_state["qa_history"][-1]
            st.markdown(latest["answer"])
            st.caption(
                f"Tokens used: {latest['input_tokens']:,} in / "
                f"{latest['output_tokens']:,} out"
            )
            with st.expander("📂 Documents included in this query"):
                if latest["docs_included"]:
                    st.write("**Included:**")
                    for fn in latest["docs_included"]:
                        st.write(f"  • {fn}")
                if latest["docs_skipped"]:
                    st.write("**Skipped (token budget):**")
                    for fn in latest["docs_skipped"]:
                        st.write(f"  • {fn}")

            if len(st.session_state["qa_history"]) > 1:
                st.markdown("---")
                st.subheader("Previous Questions")
                for i, entry in enumerate(reversed(st.session_state["qa_history"][:-1]), 1):
                    with st.expander(f"Q{i}: {entry['question'][:80]}"):
                        st.markdown(entry["answer"])

