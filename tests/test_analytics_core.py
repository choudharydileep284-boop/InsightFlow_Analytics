import io
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from analytics_core import (
    DataImportError,
    apply_filter,
    apply_filters,
    build_matplotlib_png,
    build_plotly_figure,
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
)


def test_csv_import_and_duplicate_headers():
    payload = b"name,name_2,name\nAda,1,Grace\nLin,2,\n"
    df = load_uploaded_data(payload, "people.csv")
    assert list(df.columns) == ["name", "name_2", "name_3"]
    assert len(df) == 2
    assert pd.isna(df.loc[1, "name_3"])


def test_json_array_of_objects_import():
    payload = json.dumps([{"user": {"name": "Ada"}, "score": 3}, {"user": {"name": "Lin"}, "score": 4}]).encode()
    df = load_uploaded_data(payload, "scores.json")
    assert len(df) == 2
    assert "user.name" in df.columns
    nested = load_uploaded_data(b'{"rows":[{"id":1,"tags":["a","b"]}]}', "nested.json")
    assert nested.loc[0, "tags"] == '["a", "b"]'


def test_xlsx_roundtrip_import():
    original = pd.DataFrame({"city": ["Jaipur", "Delhi"], "qty": [2, 4]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        original.to_excel(writer, index=False)
    loaded = load_uploaded_data(buf.getvalue(), "data.xlsx")
    assert loaded.shape == (2, 2)
    assert loaded["qty"].tolist() == [2, 4]


def test_sqlite_import_read_only_table_and_query():
    buf = io.BytesIO()
    path = ":memory:"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sales (region TEXT, revenue REAL)")
    conn.executemany("INSERT INTO sales VALUES (?, ?)", [("North", 10.0), ("South", 15.0)])
    conn.commit()
    # Serialize a database through the sqlite backup API.
    target = tempfile_db = __import__("tempfile").NamedTemporaryFile(suffix=".db", delete=False)
    target.close()
    dest = sqlite3.connect(tempfile_db.name)
    conn.backup(dest)
    dest.close()
    conn.close()
    payload = open(tempfile_db.name, "rb").read()
    __import__("os").unlink(tempfile_db.name)
    assert list_sqlite_tables(payload) == ["sales"]
    df = load_uploaded_data(payload, "sales.db", sqlite_table="sales")
    assert df["revenue"].sum() == 25.0
    selected = load_uploaded_data(payload, "sales.db", sqlite_table="sales", sql_query="SELECT region, revenue FROM sales WHERE revenue > 10")
    assert selected.shape == (1, 2)
    with pytest.raises(DataImportError):
        load_uploaded_data(payload, "sales.db", sqlite_table="sales", sql_query="DELETE FROM sales")
    with pytest.raises(DataImportError):
        load_uploaded_data(payload, "sales.db", sqlite_table="sales", sql_query="PRAGMA table_info(sales)")
    with pytest.raises(DataImportError):
        load_uploaded_data(payload, "sales.db", sqlite_table="sales", sql_query="SELECT load_extension('/tmp/not-a-real-extension')")


def test_cleaning_filters_and_type_conversion():
    frame = pd.DataFrame({"value": [1.0, np.nan, 3.0, 3.0], "group": [" A ", "B", "B", "B"]})
    filled, count = fill_missing_values(frame, ["value"], "median")
    assert count == 1
    assert filled["value"].tolist() == [1.0, 3.0, 3.0, 3.0]
    de_duped, removed = drop_duplicate_rows(filled)
    assert removed == 2
    missing_drop, removed_rows = drop_missing_rows(frame, ["value"])
    assert removed_rows == 1
    filtered = apply_filter(frame, "value", "greater than", 1.5)
    assert len(filtered) == 2
    combined = apply_filters(frame, [{"column": "value", "operator": "greater than", "value": 1.5}, {"column": "group", "operator": "equals", "value": "B"}])
    assert len(combined) == 2
    converted, newly_missing = convert_column_type(pd.DataFrame({"qty": ["1", "x", "3"]}), "qty", "Number")
    assert newly_missing == 1
    assert converted["qty"].isna().sum() == 1
    nullable_int = pd.DataFrame({"qty": pd.Series([1, pd.NA, 2, 3], dtype="Int64")})
    filled_fractional, filled_count = fill_missing_values(nullable_int, ["qty"], "mean")
    assert filled_count == 1
    assert float(filled_fractional.loc[1, "qty"]) == 2.0
    dates = pd.DataFrame({"date": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-07-15"])})
    this_year = apply_filter(dates, "date", "year equals", 2026)
    assert len(this_year) == 2


def test_profile_exports_and_plot_png():
    df = make_demo_data(60)
    assert quality_summary(df)["rows"] == 60
    assert len(profile_dataframe(df)) == len(df.columns)
    assert export_csv_bytes(df).startswith(b"\xef\xbb\xbf")
    injection = export_csv_bytes(pd.DataFrame({"comment": ["=1+1", "normal"]}))
    assert b"'=1+1" in injection
    assert b"records" not in export_json_bytes(df)[:50]
    xlsx = export_excel_bytes(df)
    assert xlsx[:2] == b"PK"
    # Excel exports disable automatic formula interpretation for untrusted strings.
    import openpyxl
    injected_xlsx = export_excel_bytes(pd.DataFrame({"comment": ["=1+1"]}))
    workbook = openpyxl.load_workbook(io.BytesIO(injected_xlsx), data_only=False)
    assert workbook["Data"]["A2"].data_type == "s"
    pdf = export_pdf_report(df)
    assert pdf.startswith(b"%PDF")
    png = build_matplotlib_png(df, "Bar", x="region", y="revenue")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    # Count bars (no Y series) must be supported, and chart builders must agree.
    count_bar = build_plotly_figure(df, "Bar", x="region", y=None)
    assert len(count_bar.data) >= 1
    pie = build_plotly_figure(df, "Pie", x="region", y="revenue")
    assert pie.data[0].type == "pie"
    colored_scatter = build_matplotlib_png(df, "Scatter", x="units", y="revenue", color="region")
    assert colored_scatter.startswith(b"\x89PNG\r\n\x1a\n")
    colored_box = build_matplotlib_png(df, "Box", x="region", y="revenue", color="product")
    assert colored_box.startswith(b"\x89PNG\r\n\x1a\n")
    histogram = build_matplotlib_png(df, "Histogram", x="revenue", y="revenue")
    assert histogram.startswith(b"\x89PNG\r\n\x1a\n")
