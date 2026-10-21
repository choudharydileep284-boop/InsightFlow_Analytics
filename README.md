# InsightFlow Analytics Studio

A local-first Python data analytics web app with a modern Streamlit interface. It covers import, cleaning, typed filtering, interactive charting, profiling, statistics, and exports.

## Features that are implemented

- **Import:** CSV, `.xlsx`, `.xls`, JSON arrays/objects/JSON Lines, and uploaded SQLite `.db` / `.sqlite` files. Excel worksheets and SQLite tables can be selected. SQLite imports support an optional read-only `SELECT` / `WITH` query.
- **Safety limits:** 50 MB per upload, 1,000,000 rows, 2,000 columns and 5,000,000 cells. SQLite files are opened in read-only mode, extension loading is disabled, and SQL operations are restricted by an SQLite authorizer and a query time limit.
- **Cleaning:** smart numeric-median / categorical-mode fill, median, mean, mode, constant, forward/backward fill, drop rows with missing values, deduplication by all or selected columns, text trimming/lowercasing, type conversion, remove columns, and an editable working table.
- **Filters:** stack numeric, categorical, missing-value and datetime filters. Filters are non-destructive and combine with AND.
- **Analysis:** column-level profile, descriptive numeric statistics, IQR outlier screening, category frequency, and numeric correlation.
- **Charts:** interactive Plotly bar, line, pie, histogram, scatter, box and correlation heatmap. PNG generation uses Matplotlib and does not require Chrome/Kaleido.
- **Exports:** CSV, Excel workbook (data + column profile + quality summary), JSON, PNG, and PDF quality report with the selected chart when available.
- **Demo dataset:** a generated sales dataset with missing values, whitespace, and duplicate rows so the workflow can be explored immediately.

## Quick start (Windows)

1. Install Python 3.11, 3.12 or 3.13 from [python.org](https://www.python.org/downloads/). During setup, enable **Add Python to PATH**.
2. Extract this ZIP to a folder, then open PowerShell in that folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (normally `http://localhost:8501`). Use **Load demo dataset** from the sidebar or go to **Import data**.

## Quick start (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Load a file or demo dataset.
2. Check **Overview** for missing cells and duplicate rows.
3. Use **Cleaning** and verify the row/cell counts after each action.
4. Use **Filters** to make a non-destructive selection.
5. Use **Visualize** to configure a chart; it appears after clicking **Generate visualization**.
6. Choose **Filtered dataset** or **Full working dataset** in **Export**.

## Scope and limitations

- This build supports uploaded SQLite databases, not direct connections to PostgreSQL, MySQL, SQL Server, or cloud warehouses. Those need separate drivers, credentials and deployment-specific network controls.
- Data is held in the current Streamlit session and memory. There is no login, multi-user tenant isolation layer, durable storage, or server-side user account system in this starter build. Do not expose it publicly with sensitive datasets without adding authentication, deployment hardening, and a formal retention policy.
- File extensions and Streamlit's uploader type filter are not a complete content-security boundary. The app adds format checks, size/shape limits, read-only database handling, and constrained SQL execution, but no software can honestly promise “zero bugs.”
- For very wide/large datasets, browser tables and plots can become slow. The UI shows up to 1,000 preview rows; the table editor is limited to the first 5,000 rows; chart rendering samples up to 50,000 rows when needed. Cleaning, statistics and data exports operate on the working dataset within the row limit.
- CSV spreadsheet-formula escaping is enabled to reduce formula injection risk when the downloaded file is opened in spreadsheet software. The app does not change the values in the in-memory dataset.

## Test the core logic

Install the requirements, then run:

```bash
python -m pytest -q
```

The test module focuses on data logic, filters, file parsing and read-only SQLite query restrictions. It does not replace a live browser/UI test in your target environment.

## Source references used for the design

- Streamlit file upload guidance: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit caching behavior and the warning around unsafe pickle data: https://docs.streamlit.io/develop/concepts/architecture/caching
- Streamlit data download API: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- Streamlit interactive Plotly charts: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- pandas `read_sql_query`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_query.html
- SQLite read-only URI mode: https://www.sqlite.org/uri.html
- Python `sqlite3` authorizer: https://docs.python.org/3/library/sqlite3.html
- Plotly static export requirements: https://plotly.com/python/static-image-export/ (this project intentionally uses Matplotlib for static exports to avoid a Chrome/Kaleido dependency)
- Matplotlib figure export: https://matplotlib.org/stable/users/explain/figure.html
- ReportLab Platypus: https://docs.reportlab.com/reportlab/userguide/ch5_platypus/
