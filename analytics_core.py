"""Core data operations for InsightFlow Analytics Studio.

This module is intentionally UI-independent so the data logic can be tested without
starting Streamlit. Uploaded files are treated as untrusted input.
"""
from __future__ import annotations

import io
import csv
import json
import os
from contextlib import closing
import re
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 2_000
MAX_CELLS = 5_000_000
MAX_SQL_SECONDS = 5
MAX_EXCEL_EXPANDED_BYTES = 250 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".db", ".sqlite", ".sqlite3"}


class DataImportError(ValueError):
    """Raised when an uploaded dataset cannot safely be imported."""


def _check_size(payload: bytes) -> None:
    if not payload:
        raise DataImportError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise DataImportError(
            f"File is {len(payload) / (1024 * 1024):.1f} MB. The current upload limit is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )


def _check_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise DataImportError("The importer did not return a tabular dataset.")
    if df.shape[0] > MAX_ROWS:
        raise DataImportError(f"This dataset exceeds the {MAX_ROWS:,}-row safety limit.")
    if df.shape[1] > MAX_COLUMNS:
        raise DataImportError(f"This dataset exceeds the {MAX_COLUMNS:,}-column safety limit.")
    if df.shape[0] * df.shape[1] > MAX_CELLS:
        raise DataImportError(f"This dataset exceeds the {MAX_CELLS:,}-cell safety limit. Import fewer rows or columns.")
    if df.empty and len(df.columns) == 0:
        raise DataImportError("No table data or column headers were found in this file.")
    # Normalize headers and ensure each is unique: duplicate labels can break column operations.
    cleaned = []
    used: set[str] = set()
    for i, col in enumerate(df.columns):
        base = str(col).strip() or f"Unnamed_{i + 1}"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        cleaned.append(name)
    df = df.copy()
    df.columns = cleaned
    # Keep a tabular model: nested JSON arrays/objects become readable JSON text cells
    # instead of unhashable Python containers that break profiles, deduplication and export.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list)) else value
        )
        try:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
        except (TypeError, ValueError):
            pass
    return df.reset_index(drop=True)


def _sqlite_temp_path(payload: bytes) -> str:
    """Write bytes to an isolated temporary file for sqlite's read-only URI mode."""
    handle = tempfile.NamedTemporaryFile(prefix="insightflow_", suffix=".sqlite", delete=False)
    try:
        handle.write(payload)
        handle.flush()
        return handle.name
    finally:
        handle.close()


def _open_readonly_sqlite(path: str) -> sqlite3.Connection:
    # mode=ro prevents writes even if a query reaches an unexpected mutation path.
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2)
    conn.execute("PRAGMA query_only = ON")
    conn.enable_load_extension(False)
    return conn


def _read_authorizer(action: int, arg1: str | None, arg2: str | None,
                     database: str | None, source: str | None) -> int:
    """SQLite authorizer that only allows read-oriented SQL operations."""
    allowed_names = {
        "SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE"
    }
    allowed_codes = {
        getattr(sqlite3, name) for name in allowed_names if hasattr(sqlite3, name)
    }
    if action in allowed_codes:
        # Don't allow access to other attached schemas or temp tables.
        if action == getattr(sqlite3, "SQLITE_READ", -1) and database not in ("main", None):
            return sqlite3.SQLITE_DENY
        # Extension loading is disabled, but explicitly deny it when available.
        if action == getattr(sqlite3, "SQLITE_FUNCTION", -1):
            function_name = (arg2 or arg1 or "").casefold()
            if function_name == "load_extension":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _check_xlsx_archive(payload: bytes) -> None:
    """Limit ZIP expansion before asking a spreadsheet parser to inspect .xlsx files."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > 10_000:
                raise DataImportError("Excel workbook contains too many archive entries.")
            expanded_size = sum(item.file_size for item in entries)
            if expanded_size > MAX_EXCEL_EXPANDED_BYTES:
                raise DataImportError("Excel workbook expands beyond the 250 MB safety limit.")
            # Reject obviously malformed member names rather than trying to interpret them.
            if any(item.filename.startswith(("/", "\\")) or ".." in Path(item.filename).parts for item in entries):
                raise DataImportError("Excel workbook contains an unsafe archive path.")
    except zipfile.BadZipFile as exc:
        raise DataImportError("This is not a valid XLSX workbook archive.") from exc


def list_sqlite_tables(payload: bytes) -> list[str]:
    """Return user tables/views found in an uploaded SQLite database."""
    _check_size(payload)
    path = _sqlite_temp_path(payload)
    try:
        with closing(_open_readonly_sqlite(path)) as conn:
            rows = conn.execute(
                f"SELECT name FROM sqlite_master "
                f"WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                f"ORDER BY name LIMIT {MAX_COLUMNS + 1}"
            ).fetchall()
        if len(rows) > MAX_COLUMNS:
            raise DataImportError(f"This database contains more than {MAX_COLUMNS:,} tables/views; simplify it before importing.")
        return [str(row[0]) for row in rows]
    except sqlite3.DatabaseError as exc:
        raise DataImportError("This file is not a readable SQLite database.") from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def quote_sqlite_identifier(identifier: str) -> str:
    """Quote a SQLite table identifier safely."""
    if not isinstance(identifier, str) or not identifier:
        raise DataImportError("A valid table name is required.")
    return '"' + identifier.replace('"', '""') + '"'


def load_sqlite_table(payload: bytes, table_name: str, query: str | None = None) -> pd.DataFrame:
    """Load a table or a read-only SELECT query from SQLite with result limits."""
    _check_size(payload)
    path = _sqlite_temp_path(payload)
    try:
        with closing(_open_readonly_sqlite(path)) as conn:
            # Table names shown in the UI come from sqlite_master; still quote defensively.
            available = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if query and query.strip():
                query = query.strip()
                # A single SELECT / WITH query is accepted; sqlite's authorizer is the
                # definitive read-only guard. Wrapping applies a hard cap to returned rows.
                if not re.match(r"^(SELECT|WITH)\b", query, flags=re.IGNORECASE):
                    raise DataImportError("Only SELECT or read-only WITH queries are allowed.")
                query = query.rstrip().rstrip(";").strip()
                conn.set_authorizer(_read_authorizer)
                started = time.monotonic()
                conn.set_progress_handler(
                    lambda: 1 if time.monotonic() - started > MAX_SQL_SECONDS else 0,
                    2_000,
                )
                try:
                    probe = conn.execute(f"SELECT * FROM ({query}) AS _insightflow_query LIMIT 0")
                    result_columns = max(1, len(probe.description or []))
                    allowed_rows = min(MAX_ROWS, MAX_CELLS // result_columns)
                    df = pd.read_sql_query(
                        f"SELECT * FROM ({query}) AS _insightflow_query LIMIT {allowed_rows + 1}",
                        conn,
                    )
                except (sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
                    raise DataImportError(
                        "SQL could not be run. Use one read-only SELECT query against the uploaded database."
                    ) from exc
                finally:
                    conn.set_progress_handler(None, 0)
                if len(df) > allowed_rows:
                    raise DataImportError(f"Query result exceeds the current row/cell safety limit ({allowed_rows:,} rows for {len(df.columns):,} columns).")
            else:
                if table_name not in available:
                    raise DataImportError("That table is not present in the uploaded database.")
                column_count = len(conn.execute(f"PRAGMA table_info({quote_sqlite_identifier(table_name)})").fetchall())
                allowed_rows = min(MAX_ROWS, MAX_CELLS // max(1, column_count))
                conn.set_authorizer(_read_authorizer)
                started = time.monotonic()
                conn.set_progress_handler(
                    lambda: 1 if time.monotonic() - started > MAX_SQL_SECONDS else 0,
                    2_000,
                )
                try:
                    df = pd.read_sql_query(
                        f"SELECT * FROM {quote_sqlite_identifier(table_name)} LIMIT {allowed_rows + 1}",
                        conn,
                    )
                except (sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
                    raise DataImportError("Could not read that SQLite table.") from exc
                finally:
                    conn.set_progress_handler(None, 0)
                if len(df) > allowed_rows:
                    raise DataImportError(f"Table exceeds the current row/cell safety limit ({allowed_rows:,} rows for {len(df.columns):,} columns).")
        return _check_frame(df)
    except DataImportError:
        raise
    except sqlite3.DatabaseError as exc:
        raise DataImportError("This file is not a readable SQLite database.") from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def load_uploaded_data(
    payload: bytes,
    filename: str,
    *,
    sheet_name: str | None = None,
    delimiter: str = "Auto-detect",
    sqlite_table: str | None = None,
    sql_query: str | None = None,
) -> pd.DataFrame:
    """Import CSV, Excel, JSON, or SQLite bytes to a validated DataFrame."""
    _check_size(payload)
    ext = Path(filename or "").suffix.casefold()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataImportError(f"Unsupported file type '{ext or 'unknown'}'.")
    try:
        if ext == ".csv":
            sep_map = {"Comma (, )": ",", "Semicolon (; )": ";", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|"}
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = payload.decode("latin-1")
            if delimiter == "Auto-detect":
                sample = text[:100_000]
                try:
                    sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    candidates = [",", ";", "\t", "|"]
                    counts = {candidate: sample.count(candidate) for candidate in candidates}
                    sep = max(counts, key=counts.get) if max(counts.values(), default=0) else ","
            else:
                sep = sep_map.get(delimiter, ",")
            try:
                raw_header = next(csv.reader(io.StringIO(text), delimiter=sep))
            except StopIteration as exc:
                raise DataImportError("The CSV file has no header row.") from exc
            if len(raw_header) > MAX_COLUMNS:
                raise DataImportError(f"CSV exceeds the {MAX_COLUMNS:,}-column safety limit.")
            # Stable names for duplicate/blank headers before pandas auto-renames them.
            unique_headers = []
            used_headers: set[str] = set()
            for index, raw_name in enumerate(raw_header):
                base = str(raw_name).strip() or f"Unnamed_{index + 1}"
                candidate = base
                suffix = 2
                while candidate in used_headers:
                    candidate = f"{base}_{suffix}"
                    suffix += 1
                unique_headers.append(candidate)
                used_headers.add(candidate)
            allowed_rows = min(MAX_ROWS, max(1, MAX_CELLS // max(1, len(unique_headers))))
            df = pd.read_csv(io.StringIO(text), sep=sep, names=unique_headers, header=0, nrows=allowed_rows + 1)
            if len(df) > allowed_rows:
                raise DataImportError(f"CSV exceeds the current row/cell safety limit ({allowed_rows:,} rows for {len(unique_headers):,} columns).")
        elif ext in {".xlsx", ".xls"}:
            if ext == ".xlsx":
                _check_xlsx_archive(payload)
            book = pd.ExcelFile(io.BytesIO(payload))
            if not book.sheet_names:
                raise DataImportError("No worksheets were found in this Excel file.")
            chosen = sheet_name if sheet_name in book.sheet_names else book.sheet_names[0]
            header = pd.read_excel(book, sheet_name=chosen, nrows=0)
            if len(header.columns) > MAX_COLUMNS:
                raise DataImportError(f"Worksheet exceeds the {MAX_COLUMNS:,}-column safety limit.")
            allowed_rows = min(MAX_ROWS, max(1, MAX_CELLS // max(1, len(header.columns))))
            df = pd.read_excel(book, sheet_name=chosen, nrows=allowed_rows + 1)
            if len(df) > allowed_rows:
                raise DataImportError(f"Worksheet exceeds the current row/cell safety limit ({allowed_rows:,} rows for {len(df.columns):,} columns).")
        elif ext == ".json":
            try:
                decoded = payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise DataImportError("JSON must be encoded as UTF-8.") from exc
            try:
                obj = json.loads(decoded)
                if isinstance(obj, list):
                    df = pd.json_normalize(obj) if obj and isinstance(obj[0], dict) else pd.DataFrame({"value": obj})
                elif isinstance(obj, dict):
                    candidate = obj.get("data") if isinstance(obj.get("data"), list) else None
                    if candidate is None:
                        object_lists = [v for v in obj.values() if isinstance(v, list) and v and all(isinstance(item, dict) for item in v)]
                        if len(object_lists) == 1:
                            candidate = object_lists[0]
                    if candidate is not None:
                        df = pd.json_normalize(candidate)
                    elif all(not isinstance(v, (dict, list)) for v in obj.values()):
                        df = pd.DataFrame([obj])
                    else:
                        df = pd.json_normalize(obj)
                else:
                    df = pd.DataFrame({"value": [obj]})
            except json.JSONDecodeError:
                try:
                    df = pd.read_json(io.BytesIO(payload), lines=True, nrows=MAX_ROWS + 1)
                except (ValueError, TypeError, UnicodeDecodeError) as exc:
                    raise DataImportError("Could not parse JSON. Use a JSON array/object or JSON Lines file.") from exc
            if len(df) > MAX_ROWS:
                raise DataImportError(f"JSON exceeds the {MAX_ROWS:,}-row safety limit.")
        else:
            if not sqlite_table and not (sql_query and sql_query.strip()):
                tables = list_sqlite_tables(payload)
                if not tables:
                    raise DataImportError("No user tables or views were found in this SQLite database.")
                raise DataImportError("Select a table or enter a read-only SQL query before loading this database.")
            df = load_sqlite_table(payload, sqlite_table or "", query=sql_query)
    except DataImportError:
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataImportError(f"Could not parse {ext} data. Check the file format and delimiter.") from exc
    except ImportError as exc:
        message = "Excel .xls files require xlrd; install dependencies from requirements.txt."
        if ext == ".xls":
            raise DataImportError(message) from exc
        raise DataImportError("A required file-reading dependency is missing. Install requirements.txt.") from exc
    except Exception as exc:
        # Keep internal implementation details out of normal user-facing error strings.
        raise DataImportError(f"Import failed ({type(exc).__name__}). Check the file or selected worksheet.") from exc
    return _check_frame(df)


def make_demo_data(rows: int = 180, seed: int = 17) -> pd.DataFrame:
    """Reproducible demo sales dataset with representative cleaning issues."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    regions = rng.choice(["North", "South", "East", "West"], rows)
    products = rng.choice(["Laptop", "Phone", "Tablet", "Monitor", "Headphones"], rows)
    units = rng.integers(1, 12, rows).astype(float)
    prices = {"Laptop": 720.0, "Phone": 430.0, "Tablet": 280.0, "Monitor": 210.0, "Headphones": 85.0}
    revenue = np.array([prices[p] for p in products]) * units
    df = pd.DataFrame({
        "order_date": dates,
        "region": regions,
        "product": products,
        "units": units,
        "unit_price": [prices[p] for p in products],
        "revenue": np.round(revenue + rng.normal(0, 35, rows), 2),
        "channel": rng.choice(["Online", "Retail", "Partner"], rows),
    })
    if rows > 30:
        df.loc[[7, 29], "region"] = None
        df.loc[13, "revenue"] = np.nan
        df.loc[20, "product"] = "  Phone  "
        df.loc[30] = df.loc[4]
    return _check_frame(df)


def _safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        return int(series.map(lambda value: repr(value)).nunique(dropna=dropna))


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level quality/profile table."""
    records: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        examples = series.dropna().head(3).astype(str).tolist()
        records.append({
            "Column": str(col),
            "Type": str(series.dtype),
            "Non-empty": non_null,
            "Missing": missing,
            "Missing %": round((missing / n * 100) if n else 0.0, 2),
            "Unique": _safe_nunique(series, dropna=True),
            "Example values": " · ".join(examples)[:160],
        })
    return pd.DataFrame(records)


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    cells = rows * cols
    missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [str(c) for c in df.columns if _safe_nunique(df[c], dropna=False) <= 1]
    return {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing,
        "missing_pct": (missing / cells * 100) if cells else 0.0,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": len(numeric_cols),
        "constant_columns": constant_cols,
        "memory_mb": float(df.memory_usage(index=True, deep=True).sum() / (1024 * 1024)),
    }


def _fill_series_safely(series: pd.Series, value: Any) -> pd.Series:
    """Fill a series, promoting nullable integer columns when a fractional value is needed."""
    try:
        return series.fillna(value)
    except (TypeError, ValueError):
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").astype("Float64").fillna(value)
        raise


def fill_missing_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    method: str,
    constant: Any = "Unknown",
) -> tuple[pd.DataFrame, int]:
    """Fill NA values without mutating the caller's DataFrame."""
    result = df.copy()
    selected = [c for c in columns if c in result.columns]
    before = int(result[selected].isna().sum().sum()) if selected else 0
    method = method.casefold()
    for col in selected:
        series = result[col]
        if method == "smart (median / mode)":
            if pd.api.types.is_numeric_dtype(series):
                fill = series.median()
                if pd.isna(fill):
                    continue
            else:
                modes = series.mode(dropna=True)
                if modes.empty:
                    continue
                fill = modes.iloc[0]
            result[col] = series.fillna(fill)
        elif method == "median":
            if pd.api.types.is_numeric_dtype(series):
                value = series.median()
                if pd.notna(value):
                    result[col] = _fill_series_safely(series, value)
        elif method == "mean":
            if pd.api.types.is_numeric_dtype(series):
                value = series.mean()
                if pd.notna(value):
                    result[col] = _fill_series_safely(series, value)
        elif method == "mode":
            modes = series.mode(dropna=True)
            if not modes.empty:
                result[col] = series.fillna(modes.iloc[0])
        elif method == "constant":
            value = constant
            if pd.api.types.is_numeric_dtype(series):
                try:
                    value = pd.to_numeric(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Constant value must be numeric for column '{col}'.")
            result[col] = _fill_series_safely(series, value)
        elif method == "forward fill":
            result[col] = series.ffill()
        elif method == "backward fill":
            result[col] = series.bfill()
        else:
            raise ValueError("Unsupported missing-value fill method.")
    after = int(result[selected].isna().sum().sum()) if selected else 0
    return result, before - after


def drop_missing_rows(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, int]:
    selected = [c for c in columns if c in df.columns]
    if not selected:
        return df.copy().reset_index(drop=True), 0
    result = df.dropna(subset=selected).copy()
    return result.reset_index(drop=True), len(df) - len(result)


def drop_duplicate_rows(
    df: pd.DataFrame, columns: Iterable[str] | None = None, keep: str = "first"
) -> tuple[pd.DataFrame, int]:
    selected = [c for c in (columns or []) if c in df.columns]
    subset = selected or None
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False")
    result = df.drop_duplicates(subset=subset, keep=keep).copy().reset_index(drop=True)
    return result, len(df) - len(result)


def apply_filter(df: pd.DataFrame, column: str, operator: str, value: Any = None,
                 value2: Any = None) -> pd.DataFrame:
    """Return rows matching a single typed filter rule."""
    if column not in df.columns:
        raise ValueError("The selected filter column no longer exists.")
    series = df[column]
    op = operator.casefold()
    if op in {"is missing", "is null"}:
        mask = series.isna()
    elif op in {"is not missing", "is not null"}:
        mask = series.notna()
    elif op in {"contains", "does not contain"}:
        mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        if op == "does not contain":
            mask = ~mask
    elif op in {"equals", "is", "not equals", "is not"}:
        if op in {"not equals", "is not"}:
            mask = series.ne(value) & series.notna()
        else:
            mask = series.eq(value)
    elif op in {"greater than", "greater than or equal", "less than", "less than or equal", "between"}:
        numeric = pd.to_numeric(series, errors="coerce")
        v1 = pd.to_numeric(value, errors="coerce")
        v2 = pd.to_numeric(value2, errors="coerce") if value2 is not None else None
        if pd.isna(v1):
            raise ValueError("Enter a valid numeric filter value.")
        if op == "greater than":
            mask = numeric > v1
        elif op == "greater than or equal":
            mask = numeric >= v1
        elif op == "less than":
            mask = numeric < v1
        elif op == "less than or equal":
            mask = numeric <= v1
        else:
            if pd.isna(v2):
                raise ValueError("Enter a valid upper bound.")
            lo, hi = sorted([float(v1), float(v2)])
            mask = numeric.between(lo, hi, inclusive="both")
    elif op in {"year equals", "date on/after", "date on/before", "date between"}:
        dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
        if op == "year equals":
            year = int(value)
            mask = dt.dt.year.eq(year)
        else:
            d1 = pd.to_datetime(value, errors="coerce")
            if pd.isna(d1):
                raise ValueError("Enter a valid date.")
            if op == "date on/after":
                mask = dt >= d1
            elif op == "date on/before":
                mask = dt <= (d1 + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
            else:
                d2 = pd.to_datetime(value2, errors="coerce")
                if pd.isna(d2):
                    raise ValueError("Enter a valid end date.")
                lo, hi = sorted([d1, d2])
                mask = dt.between(lo, hi + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
    elif op == "one of" and isinstance(value, (list, tuple, set)):
        mask = series.isin(list(value))
    else:
        raise ValueError(f"Unsupported filter operator: {operator}")
    return df.loc[mask.fillna(False)].copy().reset_index(drop=True)


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = df.copy()
    for rule in filters:
        result = apply_filter(
            result,
            column=rule["column"],
            operator=rule["operator"],
            value=rule.get("value"),
            value2=rule.get("value2"),
        )
    return result


def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> tuple[pd.DataFrame, int]:
    """Convert one column and return (new frame, newly-missing value count)."""
    if column not in df.columns:
        raise ValueError("Column does not exist.")
    result = df.copy()
    original_missing = int(result[column].isna().sum())
    series = result[column]
    target = target_type.casefold()
    if target == "text":
        result[column] = series.astype("string")
    elif target == "number":
        result[column] = pd.to_numeric(series, errors="coerce")
    elif target == "integer":
        result[column] = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    elif target in {"date/time", "datetime"}:
        result[column] = pd.to_datetime(series, errors="coerce")
    elif target == "boolean":
        mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
        result[column] = series.astype("string").str.strip().str.casefold().map(mapping).astype("boolean")
    else:
        raise ValueError("Unsupported target type.")
    new_missing = max(0, int(result[column].isna().sum()) - original_missing)
    return result, new_missing


def clean_text_columns(df: pd.DataFrame, columns: Iterable[str], *, lowercase: bool = False,
                       replace_blank: bool = True) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    changed = 0
    for col in columns:
        if col not in result.columns:
            continue
        if not (pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col])):
            continue
        before = result[col].copy()
        text = result[col].astype("string").str.strip()
        if lowercase:
            text = text.str.lower()
        if replace_blank:
            text = text.replace("", pd.NA)
        changed += int((before.astype("string") != text).fillna(False).sum())
        result[col] = text
    return result, changed


def _safe_spreadsheet_text(df: pd.DataFrame) -> pd.DataFrame:
    """Escape formula-like text in CSV exports, without changing the in-app data."""
    result = df.copy()
    danger = re.compile(r"^[\s\u0000-\u001f]*[=+@-]")
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col]):
            result[col] = result[col].map(
                lambda v: ("'" + v) if isinstance(v, str) and danger.match(v) else v
            )
    return result


def export_csv_bytes(df: pd.DataFrame, *, excel_safe: bool = True) -> bytes:
    out = _safe_spreadsheet_text(df) if excel_safe else df
    return out.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def export_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", date_format="iso", force_ascii=False, indent=2).encode("utf-8")


def export_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(
        buffer,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}},
    ) as writer:
        df.to_excel(writer, sheet_name="Data", index=False, freeze_panes=(1, 0))
        profile_dataframe(df).to_excel(writer, sheet_name="Column Profile", index=False)
        if len(df.columns):
            writer.sheets["Data"].autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
        summary = quality_summary(df)
        rows = [{"Metric": key, "Value": (", ".join(value) if isinstance(value, list) else value)}
                for key, value in summary.items()]
        pd.DataFrame(rows).to_excel(writer, sheet_name="Quality Summary", index=False)
        # Set readable widths while capping oversized columns.
        for sheet_name, frame in (("Data", df), ("Column Profile", profile_dataframe(df))):
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns):
                try:
                    sample_width = int(frame[col].astype(str).head(200).map(len).quantile(0.9)) if len(frame) else 10
                except (TypeError, ValueError):
                    sample_width = 12
                width = min(max(len(str(col)) + 2, sample_width + 2, 10), 34)
                ws.set_column(idx, idx, width)
            ws.freeze_panes(1, 0)
    return buffer.getvalue()


def _escape_paragraph(value: Any) -> str:
    from xml.sax.saxutils import escape
    text = "" if pd.isna(value) else str(value)
    return escape(text[:500])


def export_pdf_report(df: pd.DataFrame, *, title: str = "InsightFlow Data Analysis Report",
                      chart_png: bytes | None = None) -> bytes:
    """Build a compact PDF report using ReportLab; chart image is optional."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

    buffer = io.BytesIO()
    page_size = landscape(A4) if len(df.columns) > 7 else A4
    doc = SimpleDocTemplate(buffer, pagesize=page_size, rightMargin=14*mm, leftMargin=14*mm,
                            topMargin=14*mm, bottomMargin=14*mm, title=title[:180])
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="IFTitle", parent=styles["Title"], textColor=colors.HexColor("#152C4F"),
                              alignment=TA_LEFT, fontSize=22, leading=27, spaceAfter=5*mm))
    styles.add(ParagraphStyle(name="IFSmall", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="IFSmallHeader", parent=styles["BodyText"], fontName="Helvetica-Bold",
                              textColor=colors.white, fontSize=7.5, leading=9))
    styles.add(ParagraphStyle(name="IFSection", parent=styles["Heading2"], textColor=colors.HexColor("#0D9488"),
                              spaceBefore=5*mm, spaceAfter=2*mm))
    summary = quality_summary(df)
    story = [
        Paragraph(_escape_paragraph(title), styles["IFTitle"]),
        Paragraph("Generated by InsightFlow Analytics Studio", styles["BodyText"]),
        Paragraph(pd.Timestamp.now().strftime("%d %B %Y, %H:%M"), styles["IFSmall"]),
        Spacer(1, 5*mm),
        Paragraph("Dataset snapshot", styles["IFSection"]),
    ]
    metric_data = [
        ["Rows", "Columns", "Missing cells", "Duplicate rows", "Numeric columns", "Memory (MB)"],
        [f"{summary['rows']:,}", str(summary["columns"]), f"{summary['missing_cells']:,}",
         f"{summary['duplicate_rows']:,}", str(summary["numeric_columns"]), f"{summary['memory_mb']:.2f}"],
    ]
    mt = Table(metric_data, repeatRows=1, hAlign="LEFT")
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#152C4F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EDF4FA")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E1EC")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([mt, Spacer(1, 4*mm)])
    profile = profile_dataframe(df)
    story.append(Paragraph("Column quality profile", styles["IFSection"]))
    header = [Paragraph("Column", styles["IFSmallHeader"]), Paragraph("Type", styles["IFSmallHeader"]),
              Paragraph("Non-empty", styles["IFSmallHeader"]), Paragraph("Missing", styles["IFSmallHeader"]),
              Paragraph("Missing %", styles["IFSmallHeader"]), Paragraph("Unique", styles["IFSmallHeader"]),
              Paragraph("Examples", styles["IFSmallHeader"])]
    pdata = [header]
    for _, row in profile.head(80).iterrows():
        pdata.append([
            Paragraph(_escape_paragraph(row["Column"]), styles["IFSmall"]),
            Paragraph(_escape_paragraph(row["Type"]), styles["IFSmall"]),
            str(row["Non-empty"]), str(row["Missing"]), f"{row['Missing %']:.1f}%", str(row["Unique"]),
            Paragraph(_escape_paragraph(row["Example values"]), styles["IFSmall"]),
        ])
    available_width = page_size[0] - 28*mm
    col_widths = [available_width * x for x in (0.17, 0.13, 0.12, 0.09, 0.10, 0.08, 0.31)]
    pt = Table(pdata, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#152C4F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D7E1EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9FC")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)
    if len(profile) > 80:
        story.append(Paragraph(f"Profile table limited to the first 80 of {len(profile)} columns.", styles["IFSmall"]))
    if chart_png:
        story.append(PageBreak())
        story.append(Paragraph("Selected visualization", styles["IFSection"]))
        try:
            img = Image(io.BytesIO(chart_png))
            max_w = page_size[0] - 35*mm
            max_h = page_size[1] - 55*mm
            scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            story.append(img)
        except Exception:
            story.append(Paragraph("The selected chart could not be embedded in this report.", styles["IFSmall"]))
    doc.build(story)
    return buffer.getvalue()


def _chart_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Limit render work without changing the underlying dataset or exports.
    if len(df) > 50_000:
        return df.sample(50_000, random_state=42)
    return df


def build_plotly_figure(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    aggregation: str = "Sum",
    top_n: int = 20,
    bins: int = 30,
    title: str | None = None,
):
    """Create an interactive Plotly figure from the supplied chart settings."""
    import plotly.express as px
    from plotly.subplots import make_subplots
    from plotly import graph_objects as go

    data = _chart_frame(df.copy())
    kind = chart_type.casefold()
    if data.empty:
        raise ValueError("There are no rows to chart after filtering.")
    if kind == "correlation heatmap":
        numeric_all = data.select_dtypes(include=[np.number])
        if numeric_all.shape[1] < 2:
            raise ValueError("A correlation heatmap requires at least two numeric columns.")
        numeric = numeric_all.iloc[:, :50]
        corr = numeric.corr(numeric_only=True)
        heatmap_title = title or ("Numeric correlation matrix" if numeric_all.shape[1] <= 50 else "Numeric correlation matrix (first 50 columns)")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title=heatmap_title, aspect="auto")
    else:
        if not x or x not in data.columns:
            raise ValueError("Choose a valid X/category column.")
        if kind in {"line", "scatter", "box"} and (not y or y not in data.columns):
            raise ValueError("Choose a valid Y/value column.")
        if kind == "bar" and y is not None and y not in data.columns:
            raise ValueError("Choose a valid Y/value column.")
        if kind in {"bar", "line", "pie"}:
            if kind == "pie" and y is not None and y in data.columns and pd.api.types.is_numeric_dtype(data[y]):
                agg = aggregation.casefold()
                if agg == "mean":
                    plot_df = data.groupby(x, dropna=False, observed=False)[y].mean().reset_index()
                elif agg == "count":
                    plot_df = data.groupby(x, dropna=False, observed=False)[y].count().reset_index(name="Count")
                    y = "Count"
                else:
                    plot_df = data.groupby(x, dropna=False, observed=False)[y].sum().reset_index()
                plot_df[x] = plot_df[x].astype("string").fillna("(missing)")
                plot_df = plot_df.sort_values(y, ascending=False).head(max(1, int(top_n)))
                fig = px.pie(plot_df, names=x, values=y, title=title or f"{aggregation} {y} by {x}", hole=0.42)
            elif kind == "pie" or y is None:
                grouped = data[x].astype("string").fillna("(missing)").value_counts().head(max(1, int(top_n)))
                plot_df = grouped.rename_axis(x).reset_index(name="Count")
                if kind == "pie":
                    fig = px.pie(plot_df, names=x, values="Count", title=title or f"Share by {x}", hole=0.42)
                else:
                    fig = px.bar(plot_df, x=x, y="Count", title=title or f"Count by {x}", text_auto=".3s")
            else:
                if pd.api.types.is_numeric_dtype(data[y]):
                    agg = aggregation.casefold()
                    if agg == "mean":
                        plot_df = data.groupby(x, dropna=False, observed=False)[y].mean().reset_index()
                    elif agg == "count":
                        plot_df = data.groupby(x, dropna=False, observed=False)[y].count().reset_index(name="Count")
                        y = "Count"
                    else:
                        plot_df = data.groupby(x, dropna=False, observed=False)[y].sum().reset_index()
                    plot_df[x] = plot_df[x].astype("string").fillna("(missing)")
                    plot_df = plot_df.sort_values(y, ascending=(kind == "bar")).head(max(1, int(top_n)))
                else:
                    plot_df = data.groupby(x, dropna=False, observed=False)[y].count().reset_index(name="Count")
                    y = "Count"
                    plot_df[x] = plot_df[x].astype("string").fillna("(missing)")
                    plot_df = plot_df.sort_values(y, ascending=(kind == "bar")).head(max(1, int(top_n)))
                if kind == "bar":
                    fig = px.bar(plot_df, x=x, y=y, title=title or f"{aggregation} {y} by {x}", text_auto=".3s")
                else:
                    try:
                        plot_df = plot_df.sort_values(x)
                    except TypeError:
                        pass
                    fig = px.line(plot_df, x=x, y=y, markers=True, title=title or f"{y} over {x}")
        elif kind == "histogram":
            target = y if y in data.columns and pd.api.types.is_numeric_dtype(data[y]) else x
            if not target or not pd.api.types.is_numeric_dtype(data[target]):
                raise ValueError("Choose a numeric column for the histogram.")
            fig = px.histogram(data, x=target, nbins=max(5, min(int(bins), 150)), color=color if color in data.columns else None,
                               title=title or f"Distribution of {target}")
        elif kind == "scatter":
            if not pd.api.types.is_numeric_dtype(data[x]) or not pd.api.types.is_numeric_dtype(data[y]):
                raise ValueError("Scatter plots require numeric X and Y columns.")
            fig = px.scatter(data, x=x, y=y, color=color if color in data.columns else None,
                             opacity=0.78, title=title or f"{y} vs {x}", trendline=None)
        elif kind == "box":
            if not pd.api.types.is_numeric_dtype(data[y]):
                raise ValueError("Box plots require a numeric Y column.")
            fig = px.box(data, x=x, y=y, color=color if color in data.columns else None,
                         points="outliers", title=title or f"Distribution of {y} by {x}")
        else:
            raise ValueError("Unsupported chart type.")
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=22, r=22, t=65, b=28),
        font=dict(family="Inter, Arial, sans-serif", color="#19304D"),
        title=dict(x=0.02, xanchor="left", font=dict(size=18)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E7EDF4", zeroline=False)
    return fig


def build_matplotlib_png(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    aggregation: str = "Sum",
    top_n: int = 20,
    bins: int = 30,
    title: str | None = None,
) -> bytes:
    """Render a high-resolution PNG without requiring Chrome/Kaleido."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _chart_frame(df.copy())
    if data.empty:
        raise ValueError("There are no rows to chart after filtering.")
    kind = chart_type.casefold()
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")
    ax.grid(axis="y", color="#E5EAF1", linewidth=0.8)
    ax.set_axisbelow(True)
    if kind == "correlation heatmap":
        numeric_all = data.select_dtypes(include=[np.number])
        if numeric_all.shape[1] < 2:
            plt.close(fig)
            raise ValueError("A correlation heatmap requires at least two numeric columns.")
        numeric = numeric_all.iloc[:, :50]
        corr = numeric.corr(numeric_only=True)
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > 0.55 else "#1C2F47")
        fig.colorbar(image, ax=ax, shrink=0.78, label="Correlation")
    else:
        if kind != "histogram" and (not x or x not in data.columns):
            plt.close(fig)
            raise ValueError("Choose a valid X/category column.")
        if kind in {"line", "scatter", "box"} and (not y or y not in data.columns):
            plt.close(fig)
            raise ValueError("Choose a valid Y/value column.")
        if kind == "bar" and y is not None and y not in data.columns:
            plt.close(fig)
            raise ValueError("Choose a valid Y/value column.")
        if kind in {"bar", "line", "pie"}:
            if kind == "pie" and y is not None and y in data.columns and pd.api.types.is_numeric_dtype(data[y]):
                agg = aggregation.casefold()
                if agg == "mean":
                    grouped = data.groupby(x, dropna=False, observed=False)[y].mean()
                elif agg == "count":
                    grouped = data.groupby(x, dropna=False, observed=False)[y].count()
                else:
                    grouped = data.groupby(x, dropna=False, observed=False)[y].sum()
                grouped.index = grouped.index.map(lambda z: "(missing)" if pd.isna(z) else str(z))
                grouped = grouped.sort_values(ascending=False).head(max(1, int(top_n)))
                ax.clear()
                ax.pie(grouped.values, labels=grouped.index, autopct="%1.1f%%", startangle=90,
                       wedgeprops={"width": 0.56, "edgecolor": "white", "linewidth": 1.2})
                ax.axis("equal")
            elif kind == "pie" or y is None:
                counts = data[x].astype("string").fillna("(missing)").value_counts().head(max(1, int(top_n)))
                labels, values = counts.index.astype(str), counts.values
                if kind == "pie":
                    ax.clear()
                    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                           wedgeprops={"width": 0.56, "edgecolor": "white", "linewidth": 1.2})
                    ax.axis("equal")
                else:
                    ax.bar(labels.astype(str), values)
                    ax.tick_params(axis="x", rotation=35)
            else:
                series = data[y]
                if pd.api.types.is_numeric_dtype(series):
                    if aggregation.casefold() == "mean":
                        grouped = data.groupby(x, dropna=False, observed=False)[y].mean()
                    elif aggregation.casefold() == "count":
                        grouped = data.groupby(x, dropna=False, observed=False)[y].count()
                    else:
                        grouped = data.groupby(x, dropna=False, observed=False)[y].sum()
                else:
                    grouped = data.groupby(x, dropna=False, observed=False)[y].count()
                grouped.index = grouped.index.map(lambda z: "(missing)" if pd.isna(z) else str(z))
                grouped = grouped.sort_values(ascending=(kind == "bar")).head(max(1, int(top_n)))
                if kind == "bar":
                    ax.bar(grouped.index.astype(str), grouped.values)
                    ax.tick_params(axis="x", rotation=35)
                else:
                    grouped = grouped.sort_index()
                    ax.plot(grouped.index.astype(str), grouped.values, marker="o", linewidth=2.2)
                    ax.tick_params(axis="x", rotation=35)
        elif kind == "histogram":
            target = y if y in data.columns and pd.api.types.is_numeric_dtype(data[y]) else x
            if not target or not pd.api.types.is_numeric_dtype(data[target]):
                plt.close(fig)
                raise ValueError("Choose a numeric column for the histogram.")
            if color and color in data.columns and not pd.api.types.is_numeric_dtype(data[color]):
                color_values = data[color].astype("string").fillna("(missing)")
                top_values = color_values.value_counts().head(8).index.tolist()
                categories = [(v, color_values == v) for v in top_values]
                other_mask = ~color_values.isin(top_values)
                if other_mask.any():
                    categories.append(("Other", other_mask))
                for label, mask in categories:
                    vals = pd.to_numeric(data.loc[mask, target], errors="coerce").dropna()
                    if not vals.empty:
                        ax.hist(vals, bins=max(5, min(int(bins), 150)), alpha=0.54,
                                edgecolor="white", linewidth=0.6, label=str(label))
                if categories:
                    ax.legend(title=color, fontsize=8)
            else:
                vals = pd.to_numeric(data[target], errors="coerce").dropna()
                ax.hist(vals, bins=max(5, min(int(bins), 150)), edgecolor="white", linewidth=0.8)
            ax.set_xlabel(target)
            ax.set_ylabel("Count")
        elif kind == "scatter":
            if not pd.api.types.is_numeric_dtype(data[x]) or not y or not pd.api.types.is_numeric_dtype(data[y]):
                plt.close(fig)
                raise ValueError("Scatter plots require numeric X and Y columns.")
            if color and color in data.columns and pd.api.types.is_numeric_dtype(data[color]):
                color_vals = pd.to_numeric(data[color], errors="coerce")
                sc = ax.scatter(data[x], data[y], c=color_vals, cmap="viridis", alpha=0.72, s=26)
                fig.colorbar(sc, ax=ax, label=color)
            elif color and color in data.columns:
                color_values = data[color].astype("string").fillna("(missing)")
                top_values = color_values.value_counts().head(10).index.tolist()
                for label in top_values:
                    mask = color_values == label
                    ax.scatter(data.loc[mask, x], data.loc[mask, y], alpha=0.7, s=24, label=str(label))
                other_mask = ~color_values.isin(top_values)
                if other_mask.any():
                    ax.scatter(data.loc[other_mask, x], data.loc[other_mask, y], alpha=0.6, s=24, label="Other")
                ax.legend(title=color, fontsize=8)
            else:
                ax.scatter(data[x], data[y], alpha=0.65, s=24)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif kind == "box":
            if not y or not pd.api.types.is_numeric_dtype(data[y]):
                plt.close(fig)
                raise ValueError("Box plots require a numeric Y column.")
            groups = []
            labels = []
            if color and color in data.columns and x and color != x:
                grouped_iter = data.groupby([x, color], dropna=False, observed=False)
                for names, chunk in grouped_iter:
                    if len(labels) >= 20:
                        break
                    groups.append(pd.to_numeric(chunk[y], errors="coerce").dropna().to_numpy())
                    label_bits = ["(missing)" if pd.isna(name) else str(name) for name in names]
                    labels.append(" · ".join(label_bits))
            elif x:
                for name, chunk in data.groupby(x, dropna=False, observed=False):
                    if len(labels) >= 20:
                        break
                    groups.append(pd.to_numeric(chunk[y], errors="coerce").dropna().to_numpy())
                    labels.append("(missing)" if pd.isna(name) else str(name))
            else:
                groups = [pd.to_numeric(data[y], errors="coerce").dropna().to_numpy()]
                labels = [y]
            ax.boxplot(groups, tick_labels=labels, patch_artist=True)
            ax.tick_params(axis="x", rotation=35)
            ax.set_ylabel(y)
        else:
            plt.close(fig)
            raise ValueError("Unsupported chart type.")
    ax.set_title(title or f"{chart_type.title()} analysis", loc="left", pad=16, fontsize=15, color="#173052")
    if kind not in {"pie", "correlation heatmap"}:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D6DFE9")
        ax.spines["bottom"].set_color("#D6DFE9")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()
