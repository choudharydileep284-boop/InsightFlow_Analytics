from __future__ import annotations

import io
import json
import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics_core import (
    MAX_UPLOAD_BYTES,
    DataImportError,
    apply_filter,
    apply_filters,
    build_matplotlib_png,
    build_plotly_figure,
    clean_text_columns,
    convert_column_type,
    drop_duplicate_rows,
    drop_missing_rows,
    export_csv_bytes,
    export_excel_bytes,
    export_json_bytes,
    export_pdf_report,
    fill_missing_values,
    list_sqlite_tables,
    load_uploaded_data,
    make_demo_data,
    profile_dataframe,
    quality_summary,
    quote_sqlite_identifier,
)

st.set_page_config(
    page_title="InsightFlow | Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS is scoped to Streamlit components; no user-supplied value is rendered as HTML.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#142640; --muted:#66758B; --line:#E3EAF2; --teal:#0F9D91; --blue:#3267D6; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg, #F7FAFE 0%, #F2F6FB 100%); color: var(--ink); }
    [data-testid="stSidebar"] { background: #101E33; border-right: 1px solid #243650; }
    [data-testid="stSidebar"] * { color: #D9E5F3; }
    [data-testid="stSidebar"] .stRadio label { padding: 7px 10px; border-radius: 9px; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.07); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #A9B8CC; }
    .brand-lockup { display:flex; align-items:center; gap:11px; padding: 7px 2px 19px 2px; }
    .brand-mark { display:flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:13px; background:linear-gradient(135deg,#20B7AA,#4178EF); color:#fff; font-size:21px; box-shadow:0 8px 22px #0a172e66; }
    .brand-title { color:#F5F8FC; font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; letter-spacing:-.4px; }
    .brand-sub { color:#8FA2BC; font-size:11px; margin-top:1px; }
    .hero { background:linear-gradient(115deg,#102641 0%,#153C60 58%,#116F78 120%); border-radius:22px; padding:28px 30px; color:white; position:relative; overflow:hidden; margin-bottom:20px; box-shadow:0 14px 34px #14264014; }
    .hero:after { content:''; position:absolute; right:-60px; top:-100px; width:300px; height:300px; border-radius:50%; border:1px solid #ffffff20; box-shadow:0 0 0 34px #ffffff08,0 0 0 70px #ffffff05; }
    .hero-eyebrow { font-size:11px; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; color:#7CE0D4; margin-bottom:9px; }
    .hero h1 { color:white; font-family:'Space Grotesk',sans-serif; font-size:31px; line-height:1.15; letter-spacing:-1px; margin:0 0 9px 0; }
    .hero p { color:#CAD9E8; font-size:14px; margin:0; max-width:760px; line-height:1.6; }
    .section-title { font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:700; color:#142640; letter-spacing:-.4px; margin:5px 0 3px; }
    .section-sub { color:#738196; font-size:13px; margin-bottom:15px; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:15px; padding:16px 17px; min-height:102px; box-shadow:0 5px 16px #1b315008; }
    .metric-label { color:#697A91; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.7px; }
    .metric-value { color:#152B47; font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; line-height:1.35; margin-top:5px; }
    .metric-note { color:#91A0B2; font-size:11px; margin-top:1px; }
    .panel { background:white; border:1px solid var(--line); border-radius:16px; padding:17px 18px; box-shadow:0 5px 16px #1b315008; }
    .soft-note { color:#697A91; font-size:12px; line-height:1.55; }
    div[data-testid="stDataFrame"] { border:1px solid #E4EAF2; border-radius:12px; overflow:hidden; }
    div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:600; border:1px solid #D8E2ED; min-height:40px; }
    div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] { background:linear-gradient(120deg,#176C82,#138E91); color:white; border:0; }
    div[data-testid="stForm"] { background:white; border:1px solid var(--line); border-radius:14px; padding:16px; }
    [data-testid="stMetric"] { background:white; padding:15px 16px; border:1px solid var(--line); border-radius:13px; }
    .status-pill { border-radius:30px; padding:5px 10px; background:#E3F7F2; color:#087D72; font-size:11px; font-weight:700; display:inline-block; }
    .empty-state { background:white; border:1px dashed #CBD7E5; border-radius:17px; padding:28px 24px; text-align:center; }
    .empty-icon { font-size:32px; margin-bottom:8px; }
    .empty-title { color:#19314E; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:20px; }
    .empty-copy { color:#74849A; font-size:13px; max-width:600px; margin:7px auto 0; line-height:1.6; }
    [data-testid="stAlert"] { border-radius:12px; }
    @media (max-width: 700px) { .hero { padding:22px 18px; } .hero h1 { font-size:25px; } .metric-value { font-size:23px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Session state ---------------------------------------------------------
DEFAULTS = {
    "df": None,
    "source_df": None,
    "source_name": "",
    "loaded_at": None,
    "filters": [],
    "operation_log": [],
    "pending_upload": None,
    "chart_png": None,
    "last_chart_spec": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

PAGES = ["Import data", "Overview", "Cleaning", "Filters", "Visualize", "Insights", "Export"]


def set_dataset(df: pd.DataFrame, source_name: str) -> None:
    st.session_state.df = df.copy().reset_index(drop=True)
    # Cleaning operations return new frames rather than mutating this source object.
    st.session_state.source_df = st.session_state.df
    st.session_state.source_name = source_name
    st.session_state.loaded_at = pd.Timestamp.now().strftime("%d %b %Y, %H:%M")
    st.session_state.filters = []
    st.session_state.operation_log = [f"Imported {len(df):,} rows × {len(df.columns):,} columns from {source_name}"]
    st.session_state.chart_png = None
    st.session_state.last_chart_spec = None
    for state_key in ("current_plotly_figure", "current_plotly_json", "table_editor"):
        if state_key in st.session_state:
            del st.session_state[state_key]


def get_filtered_df() -> pd.DataFrame | None:
    df = st.session_state.df
    if df is None:
        return None
    try:
        return apply_filters(df, st.session_state.filters)
    except (ValueError, TypeError, KeyError):
        return df.copy()


def log_operation(message: str) -> None:
    st.session_state.operation_log.append(f"{pd.Timestamp.now().strftime('%H:%M:%S')} · {message}")
    st.session_state.chart_png = None
    st.session_state.last_chart_spec = None
    for state_key in ("current_plotly_figure", "current_plotly_json"):
        if state_key in st.session_state:
            del st.session_state[state_key]


def section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_no_data() -> None:
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">🗂️</div>'
        '<div class="empty-title">Start with a dataset</div>'
        '<div class="empty-copy">Upload CSV, Excel, JSON or a SQLite database, or load the built-in sample. '
        'Your data stays in this app session; there is no external AI service or account required.</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        if st.button("Load sample sales data", type="primary", use_container_width=True, key="empty_demo"):
            set_dataset(make_demo_data(), "Demo sales dataset")
            st.rerun()
    with right:
        st.caption("For file import, open **Import data** in the left navigation.")


def get_current_df() -> pd.DataFrame | None:
    df = st.session_state.df
    if df is None:
        return None
    return get_filtered_df()


def excel_sheet_names(payload: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(payload)).sheet_names


# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">✦</div><div>'
        '<div class="brand-title">InsightFlow</div><div class="brand-sub">ANALYTICS STUDIO · LOCAL FIRST</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='font-size:10px;color:#7187A2;font-weight:700;letter-spacing:1.4px;margin:6px 0 7px;'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio("Workspace navigation", PAGES, label_visibility="collapsed", key="active_page")
    st.markdown("<div style='height:1px;background:#263851;margin:18px 0;'></div>", unsafe_allow_html=True)
    if st.button("＋ Load demo dataset", use_container_width=True, key="sidebar_demo"):
        set_dataset(make_demo_data(), "Demo sales dataset")
        st.rerun()
    if st.session_state.df is not None:
        st.markdown("<div style='margin-top:16px;font-size:10px;letter-spacing:1.2px;font-weight:700;color:#7187A2;'>CURRENT DATASET</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px;color:#F3F7FB;font-weight:600;margin-top:6px;word-break:break-word;'>{st.session_state.df.shape[0]:,} rows · {st.session_state.df.shape[1]:,} columns</div>", unsafe_allow_html=True)
        safe_source = html.escape(str(st.session_state.source_name)[:90])
        st.markdown(f"<div style='font-size:11px;color:#8EA1BA;margin-top:4px;word-break:break-word;'>{safe_source}</div>", unsafe_allow_html=True)
        if st.button("Reset to imported data", help="Discard cleaning edits and clear filters, restoring the dataset as it was when loaded.", use_container_width=True, key="reset_demo"):
            if st.session_state.source_df is not None:
                st.session_state.df = st.session_state.source_df.copy().reset_index(drop=True)
                st.session_state.filters = []
                st.session_state.operation_log.append("Reset working dataset to imported source")
                st.session_state.chart_png = None
                st.session_state.last_chart_spec = None
                for state_key in ("table_editor", "current_plotly_figure", "current_plotly_json"):
                    if state_key in st.session_state:
                        del st.session_state[state_key]
            st.rerun()
    st.markdown("<div style='height:1px;background:#263851;margin:18px 0 11px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#8EA1BA;line-height:1.7;'>CSV · Excel · JSON · SQLite<br>Interactive charts · Cleaning · PDF / XLSX / PNG</div>", unsafe_allow_html=True)


# ---- Top identity / page renderer -----------------------------------------
if page == "Import data":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">DATA WORKSPACE · 01 / IMPORT</div>'
        '<h1>Bring your data into focus.</h1>'
        '<p>Import a flat file or a SQLite database, preview the structure, then move from raw records to a clean, explainable analysis.</p></div>',
        unsafe_allow_html=True,
    )
    section_heading("Import a dataset", "Files are processed in this Streamlit session. File extensions are checked, and row/size limits guard against accidental oversized imports.")
    uploader = st.file_uploader(
        "Choose a source file",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite", "sqlite3"],
        max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
        help="Supported: .csv, .xlsx, .xls, .json, .db, .sqlite, .sqlite3. Current maximum: 50 MB.",
        key="data_import_uploader",
    )
    if uploader is not None:
        upload_bytes = uploader.getvalue()
        st.session_state.pending_upload = {"name": uploader.name, "bytes": upload_bytes}
    pending = st.session_state.pending_upload
    if pending:
        pending_name = pending["name"]
        pending_bytes = pending["bytes"]
        ext = Path(pending_name).suffix.casefold()
        st.markdown(f"**Selected:** `{pending_name}` · {len(pending_bytes) / (1024*1024):.2f} MB")
        delimiter = "Auto-detect"
        sheet_name = None
        sqlite_table = None
        sql_query = None
        tables = []
        import_error = None
        if ext == ".csv":
            delimiter = st.selectbox("CSV delimiter", ["Auto-detect", "Comma (, )", "Semicolon (;) ", "Tab", "Pipe (|)"], index=0)
            if delimiter == "Semicolon (; )":
                delimiter = "Semicolon (;)"
        elif ext in {".xlsx", ".xls"}:
            try:
                sheets = excel_sheet_names(pending_bytes)
                sheet_name = st.selectbox("Worksheet", sheets, help="Only the selected worksheet is imported.")
            except Exception as exc:
                import_error = "Could not inspect the Excel file. Check that it is a valid workbook and that dependencies are installed."
        elif ext in {".db", ".sqlite", ".sqlite3"}:
            try:
                tables = list_sqlite_tables(pending_bytes)
                if not tables:
                    import_error = "No user tables or views were found in this SQLite database."
                else:
                    sqlite_table = st.selectbox("Table or view", tables, help="Loaded in read-only mode.")
                    use_query = st.checkbox("Use a custom read-only SQL query", value=False)
                    if use_query:
                        default_query = f"SELECT * FROM {quote_sqlite_identifier(sqlite_table)} LIMIT 1000"
                        sql_query = st.text_area(
                            "SQL query",
                            value=default_query,
                            height=105,
                            help="Use one SELECT or read-only WITH query. Writes, attached databases and PRAGMA statements are blocked.",
                        )
            except DataImportError as exc:
                import_error = str(exc)
        if import_error:
            st.error(import_error)
        load_col, info_col = st.columns([1, 2])
        with load_col:
            load_clicked = st.button("Load dataset →", type="primary", use_container_width=True, disabled=bool(import_error))
        with info_col:
            st.markdown('<div class="soft-note">Imports do not modify your original file. The working table is kept in session memory only and is not persisted to a server database.</div>', unsafe_allow_html=True)
        if load_clicked:
            try:
                loaded = load_uploaded_data(
                    pending_bytes,
                    pending_name,
                    sheet_name=sheet_name,
                    delimiter=delimiter,
                    sqlite_table=sqlite_table,
                    sql_query=sql_query,
                )
                set_dataset(loaded, pending_name + (f" · {sheet_name}" if sheet_name else "") + (f" · {sqlite_table}" if sqlite_table and not sql_query else ""))
                st.success(f"Loaded {len(loaded):,} rows and {len(loaded.columns):,} columns.")
                st.rerun()
            except (DataImportError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Import failed ({type(exc).__name__}). Check the file format and try again.")
    else:
        st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
        st.info("Choose a file above, or use **Load demo dataset** in the sidebar to explore the complete workflow without uploading anything.")
    with st.expander("Supported sources and limits", expanded=False):
        st.markdown("""
        - **CSV:** delimiter auto-detection or comma, semicolon, tab and pipe.
        - **Excel:** `.xlsx` and `.xls`; select one worksheet per import.
        - **JSON:** arrays of objects, objects, nested JSON and JSON Lines.
        - **SQL:** uploaded SQLite `.db` / `.sqlite` files only. Tables are read-only; optional custom query is restricted to read operations.
        - **Limits:** 50 MB per file, 1,000,000 rows, 2,000 columns and 5,000,000 cells. These are conservative app-level limits, not claims about the underlying libraries.
        - **Not included:** direct PostgreSQL/MySQL/cloud database connections; they require a separately configured driver and credential-management policy.
        """)

elif page == "Overview":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">WORKSPACE · OVERVIEW</div>'
        '<h1>See the shape of your data.</h1>'
        '<p>Get a quality snapshot, inspect records, and catch missing values, duplicates, and data type issues before analysis.</p></div>',
        unsafe_allow_html=True,
    )
    df = st.session_state.df
    if df is None:
        show_no_data()
    else:
        filtered = get_filtered_df()
        summary = quality_summary(df)
        filtered_rows = len(filtered)
        m = st.columns(4)
        metrics = [
            ("ROWS", f"{len(df):,}", f"{filtered_rows:,} rows after filters"),
            ("COLUMNS", f"{df.shape[1]:,}", f"{summary['numeric_columns']} numeric columns"),
            ("MISSING CELLS", f"{summary['missing_cells']:,}", f"{summary['missing_pct']:.2f}% of all cells"),
            ("DUPLICATE ROWS", f"{summary['duplicate_rows']:,}", f"{summary['memory_mb']:.2f} MB estimated in memory"),
        ]
        for col, data in zip(m, metrics):
            with col:
                metric_card(*data)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.4, 1])
        with left:
            section_heading("Data preview", "First 1,000 rows are shown here for responsiveness; exports include the full selected dataset.")
            st.dataframe(filtered.head(1000), width="stretch", height=390, hide_index=True)
            st.caption(f"Showing {min(1000, len(filtered)):,} of {len(filtered):,} filtered rows · Source: {st.session_state.source_name}")
        with right:
            section_heading("Column quality", "Missing cells, unique values, and data types.")
            profile = profile_dataframe(df)
            st.dataframe(profile, width="stretch", height=390, hide_index=True)
        with st.expander("Quick edit table values", expanded=False):
            st.caption("Edits are applied to the in-session working copy when you click Save edits. They do not change the uploaded file.")
            edited_df = st.data_editor(df.head(5000), num_rows="dynamic", width="stretch", height=360, key="table_editor")
            if len(df) > 5000:
                st.warning("The editor is capped to the first 5,000 rows. Use Cleaning for whole-dataset operations.")
            if st.button("Save edits to working dataset", type="primary", key="save_table_edits"):
                if len(df) > 5000:
                    updated = pd.concat([edited_df, df.iloc[5000:]], ignore_index=True)
                else:
                    updated = edited_df
                st.session_state.df = updated.reset_index(drop=True)
                log_operation("Saved data editor changes")
                st.success("Working dataset updated.")
                st.rerun()
        if st.session_state.operation_log:
            with st.expander("Activity log", expanded=False):
                for item in reversed(st.session_state.operation_log[-15:]):
                    st.markdown(f"- {item}")

elif page == "Cleaning":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">PREPROCESSING · 02 / CLEAN</div>'
        '<h1>Make your dataset analysis-ready.</h1>'
        '<p>Run explicit cleaning operations against the working copy. Every action reports what changed; the original upload is never overwritten.</p></div>',
        unsafe_allow_html=True,
    )
    df = st.session_state.df
    if df is None:
        show_no_data()
    else:
        summary = quality_summary(df)
        cols = st.columns(4)
        for col, metric in zip(cols, [
            ("MISSING CELLS", f"{summary['missing_cells']:,}", "Across all columns"),
            ("DUPLICATES", f"{summary['duplicate_rows']:,}", "Exact duplicate rows"),
            ("TEXT COLUMNS", str(len(df.select_dtypes(include=["object", "string"]).columns)), "Whitespace cleanup available"),
            ("WORKING ROWS", f"{len(df):,}", "Before active filters"),
        ]):
            with col:
                metric_card(*metric)
        t1, t2, t3, t4 = st.tabs(["Missing values", "Duplicates", "Text cleanup", "Convert types / columns"])
        with t1:
            section_heading("Resolve missing values", "Choose the columns and action. Numeric and categorical strategies are applied by column type.")
            missing_cols = df.columns[df.isna().any()].tolist()
            if not missing_cols:
                st.success("No missing values detected in the working dataset.")
            with st.form("missing_value_form"):
                action = st.selectbox("Action", ["Fill missing values", "Drop rows with missing values"], key="missing_action")
                selected_missing = st.multiselect("Columns to process", list(df.columns), default=missing_cols or list(df.columns)[:1], key="missing_columns")
                method = st.selectbox("Fill strategy", ["Smart (median / mode)", "Median", "Mean", "Mode", "Constant", "Forward fill", "Backward fill"], disabled=action != "Fill missing values")
                constant_value = st.text_input("Constant fill value", value="Unknown", disabled=action != "Fill missing values" or method != "Constant")
                submitted = st.form_submit_button("Apply missing-value action", type="primary")
            if submitted:
                try:
                    if action == "Drop rows with missing values":
                        updated, changed = drop_missing_rows(df, selected_missing)
                        st.session_state.df = updated
                        log_operation(f"Dropped {changed:,} rows with missing values")
                        st.success(f"Removed {changed:,} rows. Dataset now has {len(updated):,} rows.")
                    else:
                        updated, changed = fill_missing_values(df, selected_missing, method, constant_value)
                        st.session_state.df = updated
                        log_operation(f"Filled {changed:,} missing cells using {method}")
                        st.success(f"Filled {changed:,} missing cells using {method}.")
                    st.rerun()
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        with t2:
            section_heading("Remove duplicate records", "By default, duplicates are identified across all columns. Choose a key subset when needed.")
            with st.form("duplicate_form"):
                dup_columns_mode = st.radio("Compare columns", ["All columns", "Selected columns"], horizontal=True)
                dup_subset = st.multiselect("Deduplication key columns", list(df.columns), default=list(df.columns[:1]))
                keep = st.selectbox("Which duplicate to keep", ["first", "last", "none"], format_func=lambda x: {"first":"Keep first row", "last":"Keep last row", "none":"Remove every occurrence of duplicates"}[x])
                submitted_dup = st.form_submit_button("Remove duplicates", type="primary")
            if submitted_dup:
                try:
                    updated, removed = drop_duplicate_rows(
                        df,
                        columns=dup_subset if dup_columns_mode == "Selected columns" else None,
                        keep=False if keep == "none" else keep,
                    )
                    st.session_state.df = updated
                    log_operation(f"Removed {removed:,} duplicate rows")
                    st.success(f"Removed {removed:,} rows. {len(updated):,} rows remain.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with t3:
            section_heading("Normalize text fields", "Trims leading/trailing whitespace and optionally lowercases selected string columns.")
            string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
            if not string_cols:
                st.info("No text columns found. Convert a column to Text first if needed.")
            with st.form("text_clean_form"):
                chosen_text = st.multiselect("Text columns", string_cols, default=string_cols)
                lower = st.checkbox("Convert text to lowercase", value=False)
                blanks = st.checkbox("Treat blank strings as missing values", value=True)
                submit_text = st.form_submit_button("Normalize selected text", type="primary")
            if submit_text:
                updated, changed = clean_text_columns(df, chosen_text, lowercase=lower, replace_blank=blanks)
                st.session_state.df = updated
                log_operation(f"Normalized text in {len(chosen_text)} columns ({changed:,} cell values changed)")
                st.success(f"Normalized {len(chosen_text)} columns. {changed:,} changed cell values detected.")
                st.rerun()
        with t4:
            section_heading("Convert column type", "Failed conversions become missing values. Inspect the reported count before choosing a fill strategy.")
            with st.form("convert_type_form"):
                target_col = st.selectbox("Column", list(df.columns))
                target_type = st.selectbox("Convert to", ["Text", "Number", "Integer", "Date/time", "Boolean"])
                submit_convert = st.form_submit_button("Convert column", type="primary")
            if submit_convert:
                try:
                    updated, newly_missing = convert_column_type(df, target_col, target_type)
                    st.session_state.df = updated
                    log_operation(f"Converted {target_col} to {target_type}; {newly_missing:,} values became missing")
                    st.success(f"Converted '{target_col}' to {target_type}. {newly_missing:,} non-empty values could not be converted.")
                    st.rerun()
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
            st.divider()
            with st.form("drop_columns_form"):
                remove_cols = st.multiselect("Remove columns permanently from working copy", list(df.columns), key="remove_columns")
                confirm_remove = st.checkbox("I understand this removes these columns from the working copy", value=False)
                submit_remove = st.form_submit_button("Remove selected columns", type="primary")
            if submit_remove:
                if not confirm_remove or not remove_cols:
                    st.warning("Select at least one column and confirm the action.")
                elif len(remove_cols) >= len(df.columns):
                    st.error("You cannot remove every column from the dataset.")
                else:
                    st.session_state.df = df.drop(columns=remove_cols).copy()
                    st.session_state.filters = [r for r in st.session_state.filters if r["column"] not in remove_cols]
                    log_operation(f"Removed columns: {', '.join(remove_cols)}")
                    st.success(f"Removed {len(remove_cols)} columns.")
                    st.rerun()
        st.divider()
        section_heading("Preview after cleaning", "A compact live view of the current working copy.")
        st.dataframe(st.session_state.df.head(300), width="stretch", height=310, hide_index=True)

elif page == "Filters":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">DATA EXPLORATION · 03 / FILTER</div>'
        '<h1>Ask focused questions of your rows.</h1>'
        '<p>Stack filters to narrow the working table. Filters are non-destructive and can be cleared without undoing cleaning operations.</p></div>',
        unsafe_allow_html=True,
    )
    df = st.session_state.df
    if df is None:
        show_no_data()
    else:
        rules = st.session_state.filters
        filtered = get_filtered_df()
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("SOURCE ROWS", f"{len(df):,}", "Cleaned working dataset")
        with m2: metric_card("MATCHING ROWS", f"{len(filtered):,}", "After active filters")
        with m3: metric_card("ACTIVE FILTERS", str(len(rules)), "All conditions are combined with AND")
        section_heading("Add a filter", "Choose a column and a condition. Date filters work best after converting the column to Date/time in Cleaning.")
        with st.form("filter_builder"):
            fcol = st.selectbox("Column", list(df.columns), key="filter_column")
            series = df[fcol]
            numeric = pd.api.types.is_numeric_dtype(series)
            datetime_col = pd.api.types.is_datetime64_any_dtype(series)
            if numeric:
                operators = ["greater than", "greater than or equal", "less than", "less than or equal", "equals", "not equals", "between", "is missing", "is not missing"]
            elif datetime_col:
                operators = ["year equals", "date on/after", "date on/before", "date between", "is missing", "is not missing"]
            else:
                operators = ["equals", "not equals", "contains", "does not contain", "one of", "is missing", "is not missing"]
            op = st.selectbox("Condition", operators, key="filter_operator")
            fvalue = None
            fvalue2 = None
            if op in {"is missing", "is not missing"}:
                st.caption("This condition does not need a value.")
            elif op == "between" and numeric:
                f1, f2 = st.columns(2)
                fvalue = f1.number_input("Minimum", value=float(pd.to_numeric(series, errors="coerce").min()) if series.notna().any() else 0.0, key="filter_min")
                fvalue2 = f2.number_input("Maximum", value=float(pd.to_numeric(series, errors="coerce").max()) if series.notna().any() else 1.0, key="filter_max")
            elif op == "date between":
                d1, d2 = st.columns(2)
                min_date = pd.to_datetime(series, errors="coerce").min()
                max_date = pd.to_datetime(series, errors="coerce").max()
                default_start = min_date.date() if pd.notna(min_date) else date(2026, 1, 1)
                default_end = max_date.date() if pd.notna(max_date) else date(2026, 12, 31)
                fvalue = d1.date_input("Start date", value=default_start, key="filter_date_start")
                fvalue2 = d2.date_input("End date", value=default_end, key="filter_date_end")
            elif op in {"date on/after", "date on/before"}:
                dt_min = pd.to_datetime(series, errors="coerce").min()
                fvalue = st.date_input("Date", value=dt_min.date() if pd.notna(dt_min) else date.today(), key="filter_date_one")
            elif op == "year equals":
                years = pd.to_datetime(series, errors="coerce").dt.year.dropna()
                fvalue = st.number_input("Year", min_value=1, max_value=9999, value=int(years.mode().iloc[0]) if not years.mode().empty else 2026, step=1, key="filter_year")
            elif numeric:
                valid_series = pd.to_numeric(series, errors="coerce").dropna()
                default_val = float(valid_series.median()) if not valid_series.empty else 0.0
                fvalue = st.number_input("Value", value=default_val, key="filter_number")
            elif op == "one of":
                choices = series.dropna().astype(str).value_counts().head(200).index.tolist()
                fvalue = st.multiselect("Values", choices, key="filter_set")
            else:
                choices = series.dropna().astype(str).value_counts().head(200).index.tolist()
                if choices and op in {"equals", "not equals"}:
                    fvalue = st.selectbox("Value", choices, key="filter_choice")
                else:
                    fvalue = st.text_input("Text", key="filter_text")
            submitted_filter = st.form_submit_button("＋ Add filter", type="primary")
        if submitted_filter:
            rule = {"column": fcol, "operator": op, "value": fvalue, "value2": fvalue2}
            try:
                # Validate against the current table before storing it.
                apply_filter(df, fcol, op, fvalue, fvalue2)
                st.session_state.filters.append(rule)
                log_operation(f"Added filter: {fcol} {op}")
                st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
        if rules:
            st.divider()
            section_heading("Active filters", "Conditions are applied in order and combined with AND.")
            for idx, rule in enumerate(list(rules)):
                rc, rb = st.columns([8, 1])
                v = rule.get("value")
                if isinstance(v, list): v = ", ".join(map(str, v))
                if rule.get("value2") is not None:
                    v = f"{v} → {rule['value2']}"
                rc.markdown(f"**{idx + 1}.** `{rule['column']}` · {rule['operator']} · `{v if v is not None else '—'}`")
                if rb.button("✕", key=f"remove_filter_{idx}", help="Remove this filter"):
                    removed_rule = st.session_state.filters.pop(idx)
                    log_operation(f"Removed filter: {removed_rule['column']} {removed_rule['operator']}")
                    st.rerun()
            if st.button("Clear all filters", key="clear_filters"):
                st.session_state.filters = []
                log_operation("Cleared all filters")
                st.rerun()
        st.divider()
        section_heading("Filtered result", f"Showing {min(1000, len(filtered)):,} of {len(filtered):,} matching rows.")
        st.dataframe(filtered.head(1000), width="stretch", height=390, hide_index=True)

elif page == "Visualize":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">VISUAL EXPLORATION · 04 / CHARTS</div>'
        '<h1>Turn rows into a readable story.</h1>'
        '<p>Build interactive charts with configurable fields, aggregation, and category limits. Download the selected visual as a high-resolution PNG.</p></div>',
        unsafe_allow_html=True,
    )
    df = get_current_df()
    if df is None:
        show_no_data()
    elif df.empty:
        st.warning("No rows match the active filters. Clear or adjust filters before creating a chart.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = list(df.columns)
        section_heading("Chart builder", "The selected chart uses the filtered view. Static PNG generation uses Matplotlib; on-screen exploration uses Plotly.")
        with st.form("chart_builder"):
            chart_type = st.selectbox("Visualization", ["Bar", "Line", "Pie", "Histogram", "Scatter", "Box", "Correlation heatmap"])
            c1, c2, c3 = st.columns(3)
            with c1:
                x_options = ["(row count)"] + columns
                x_value = st.selectbox("X axis / category", x_options, index=1 if columns else 0)
                x_col = None if x_value == "(row count)" else x_value
            with c2:
                y_options = ["(count rows)"] + numeric_cols
                if chart_type in {"Line", "Scatter", "Box", "Histogram"} and numeric_cols:
                    y_options = numeric_cols
                    y_default = 0
                else:
                    y_default = 0
                y_value = st.selectbox("Y axis / value", y_options, index=y_default)
                y_col = None if y_value == "(count rows)" else y_value
            with c3:
                agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count"], disabled=chart_type in {"Histogram", "Scatter", "Box", "Correlation heatmap", "Pie"})
                if chart_type == "Histogram":
                    bins = st.slider("Histogram bins", 5, 100, 30)
                else:
                    bins = 30
            c4, c5, c6 = st.columns(3)
            with c4:
                top_n = st.slider("Category limit", 5, 50, 20, help="Keeps category charts readable.")
            with c5:
                color_value = st.selectbox("Color by", ["(none)"] + columns, disabled=chart_type not in {"Histogram", "Scatter", "Box"}, help="Applies to chart types where row-level grouping is meaningful.")
                color_col = None if color_value == "(none)" else color_value
            with c6:
                title = st.text_input("Chart title (optional)", value="")
            make_chart = st.form_submit_button("Generate visualization", type="primary")
        if make_chart:
            # Never leave an older visual visible if the current chart configuration fails.
            st.session_state.chart_png = None
            for state_key in ("current_plotly_figure", "current_plotly_json"):
                if state_key in st.session_state:
                    del st.session_state[state_key]
            try:
                if chart_type == "Correlation heatmap":
                    figure = build_plotly_figure(df, "Correlation heatmap", title=title or None, bins=bins, top_n=top_n)
                    png = build_matplotlib_png(df, "Correlation heatmap", title=title or None, bins=bins, top_n=top_n)
                else:
                    # If no X column is applicable, choose the first numeric field for histogram.
                    if chart_type == "Histogram" and x_col is None:
                        if y_col is None and numeric_cols:
                            x_col = numeric_cols[0]
                    figure = build_plotly_figure(df, chart_type, x=x_col, y=y_col, color=color_col, aggregation=agg, top_n=top_n, bins=bins, title=title or None)
                    png = build_matplotlib_png(df, chart_type, x=x_col, y=y_col, color=color_col, aggregation=agg, top_n=top_n, bins=bins, title=title or None)
                st.session_state.chart_png = png
                st.session_state.last_chart_spec = {"type": chart_type, "x": x_col, "y": y_col, "aggregation": agg, "top_n": top_n, "bins": bins, "title": title or f"{chart_type} analysis"}
                st.session_state.current_plotly_json = figure.to_json()
                st.session_state.current_plotly_figure = figure
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
        if st.session_state.get("current_plotly_figure") is not None:
            st.plotly_chart(st.session_state.current_plotly_figure, width="stretch", config={"displaylogo": False, "responsive": True}, key="active_plotly_chart")
            spec = st.session_state.get("last_chart_spec") or {}
            download_cols = st.columns([1, 1, 2])
            with download_cols[0]:
                st.download_button("↓ Download chart PNG", data=st.session_state.chart_png or b"", file_name="insightflow_chart.png", mime="image/png", disabled=not bool(st.session_state.chart_png), use_container_width=True)
            with download_cols[1]:
                if st.button("Clear chart", use_container_width=True):
                    st.session_state.chart_png = None
                    st.session_state.last_chart_spec = None
                    st.session_state.current_plotly_figure = None
                    st.session_state.current_plotly_json = None
                    st.rerun()
            with download_cols[2]:
                st.caption(f"Chart: {spec.get('type', '—')} · Data rows: {len(df):,} (render sample capped at 50,000 rows where needed)")
        else:
            st.markdown('<div class="panel"><div style="font-weight:700;color:#19314E;">Choose fields, then generate a chart</div><div class="soft-note" style="margin-top:6px;">Start with a bar chart for category comparisons, a line chart for trends, a histogram for distributions, or a correlation heatmap for numeric relationships.</div></div>', unsafe_allow_html=True)

elif page == "Insights":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">STATISTICAL ANALYSIS · 05 / INSIGHTS</div>'
        '<h1>Look beyond the first chart.</h1>'
        '<p>Review distributions, correlations, category concentration, and simple IQR outlier flags. These are descriptive checks, not causal conclusions.</p></div>',
        unsafe_allow_html=True,
    )
    df = get_current_df()
    if df is None:
        show_no_data()
    elif df.empty:
        st.warning("No rows match the active filters. Clear filters to calculate statistics.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = [c for c in df.columns if c not in numeric_cols]
        section_heading("Numerical summary", "Pandas descriptive statistics with additional missingness and IQR checks.")
        if numeric_cols:
            stats = df[numeric_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.reset_index(names="Column")
            missing = df[numeric_cols].isna().sum().rename("Missing").reset_index(names="Column")
            stats = stats.merge(missing, on="Column", how="left")
            st.dataframe(stats, width="stretch", hide_index=True)
            with st.expander("Potential outliers (1.5 × IQR rule)", expanded=False):
                outlier_records = []
                for col in numeric_cols:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if s.empty:
                        continue
                    q1, q3 = s.quantile([.25, .75])
                    spread = q3 - q1
                    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
                    count = int(((s < low) | (s > high)).sum())
                    outlier_records.append({"Column": col, "Lower bound": low, "Upper bound": high, "Potential outliers": count, "Share %": round(count / len(s) * 100, 2)})
                if outlier_records:
                    st.dataframe(pd.DataFrame(outlier_records).sort_values("Potential outliers", ascending=False), width="stretch", hide_index=True)
                st.caption("IQR flags are screening signals. They do not prove a row is wrong and are not automatically removed.")
        else:
            st.info("No numeric columns available. Convert a column to Number in Cleaning to calculate numeric statistics.")
        col_left, col_right = st.columns(2)
        with col_left:
            section_heading("Category concentration", "Most frequent values in one selected categorical column.")
            if string_cols:
                chosen = st.selectbox("Category column", string_cols, key="category_profile")
                counts = df[chosen].astype("string").fillna("(missing)").value_counts().head(15).rename_axis("Value").reset_index(name="Rows")
                counts["Share %"] = (counts["Rows"] / max(1, len(df)) * 100).round(2)
                st.dataframe(counts, width="stretch", hide_index=True)
                st.plotly_chart(px.bar(counts, x="Value", y="Rows", title=f"Most common values · {chosen}").update_layout(template="plotly_white", height=330), width="stretch")
            else:
                st.info("No categorical/text columns found.")
        with col_right:
            section_heading("Correlation check", "Pearson correlation between numeric columns; sensitive to outliers and non-linear relationships.")
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr(numeric_only=True)
                st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Correlation matrix").update_layout(template="plotly_white", height=440), width="stretch")
                pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
                pairs.columns = ["Column A", "Column B", "Correlation"]
                pairs["Absolute correlation"] = pairs["Correlation"].abs()
                st.dataframe(pairs.sort_values("Absolute correlation", ascending=False).head(10), width="stretch", hide_index=True)
            else:
                st.info("At least two numeric columns are required for correlation analysis.")

elif page == "Export":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">DELIVERY · 06 / EXPORT</div>'
        '<h1>Take the analysis with you.</h1>'
        '<p>Export a cleaned table, the currently filtered view, a quality workbook, or a compact PDF report. CSV spreadsheet-formula escaping is enabled by default.</p></div>',
        unsafe_allow_html=True,
    )
    df = st.session_state.df
    if df is None:
        show_no_data()
    else:
        filtered = get_filtered_df()
        section_heading("Choose export scope", "Cleaning operations affect the working dataset. Active filters affect only the filtered export option.")
        scope = st.radio("Dataset to export", ["Filtered dataset", "Full working dataset"], horizontal=True, index=0, key="export_scope")
        export_df = filtered if scope == "Filtered dataset" else df
        q = quality_summary(export_df)
        m = st.columns(4)
        for col, val in zip(m, [
            ("EXPORT ROWS", f"{len(export_df):,}", "Selected scope"),
            ("COLUMNS", str(export_df.shape[1]), "Included fields"),
            ("MISSING CELLS", f"{q['missing_cells']:,}", "Still present in selection"),
            ("DUPLICATES", f"{q['duplicate_rows']:,}", "Exact rows in selection"),
        ]):
            with col: metric_card(*val)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel"><div style="font-weight:700;font-size:16px;color:#18314F;">📄 Data files</div><div class="soft-note" style="margin:5px 0 12px;">CSV is UTF-8 with BOM; Excel includes data, column profile and quality summary sheets; JSON is a records array.</div></div>', unsafe_allow_html=True)
            st.download_button("Download CSV", data=export_csv_bytes(export_df), file_name="insightflow_data.csv", mime="text/csv", type="primary", use_container_width=True)
            st.download_button("Download Excel workbook", data=export_excel_bytes(export_df), file_name="insightflow_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            st.download_button("Download JSON", data=export_json_bytes(export_df), file_name="insightflow_data.json", mime="application/json", use_container_width=True)
        with c2:
            st.markdown('<div class="panel"><div style="font-weight:700;font-size:16px;color:#18314F;">📊 Analysis report</div><div class="soft-note" style="margin:5px 0 12px;">PDF includes dataset metrics and a column profile. The selected chart image is added when available.</div></div>', unsafe_allow_html=True)
            chart_png = st.session_state.chart_png
            if chart_png:
                st.markdown('<div class="status-pill">Chart available · will be included in PDF</div>', unsafe_allow_html=True)
            else:
                st.caption("No chart generated yet. Generate one in Visualize to include it in the PDF.")
            try:
                pdf_bytes = export_pdf_report(export_df, chart_png=chart_png)
                st.download_button("Download PDF report", data=pdf_bytes, file_name="insightflow_report.pdf", mime="application/pdf", type="primary", use_container_width=True)
            except Exception as exc:
                st.error(f"PDF report generation failed ({type(exc).__name__}). Check that reportlab is installed.")
            if chart_png:
                st.download_button("Download current chart PNG", data=chart_png, file_name="insightflow_chart.png", mime="image/png", use_container_width=True)
        st.divider()
        section_heading("Export preview", f"First {min(250, len(export_df)):,} rows of the export selection.")
        st.dataframe(export_df.head(250), width="stretch", height=320, hide_index=True)

# Footer
st.markdown("<div style='height:20px'></div><div style='border-top:1px solid #E0E7F0;padding-top:13px;color:#8290A3;font-size:11px;'>InsightFlow Analytics Studio · Local-first session workflow · Inspect results before making business decisions</div>", unsafe_allow_html=True)
