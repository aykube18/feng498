import streamlit as st
import pandas as pd
import io
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Support System",
    page_icon="assets/icon.png" if Path("assets/icon.png").exists() else ":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* Main area */
    .main .block-container { padding-top: 1.5rem; }

    /* Section headers */
    .module-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        border-left: 5px solid #3b82f6;
        padding-left: 12px;
        margin-bottom: 1rem;
    }

    /* Upload area */
    .upload-box {
        border: 2px dashed #94a3b8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: #f8fafc;
    }

    /* Status badges */
    .badge-ok   { background:#dcfce7; color:#166534; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-warn { background:#fef9c3; color:#854d0e; border-radius:6px; padding:2px 10px; font-size:0.8rem; }
    .badge-err  { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 10px; font-size:0.8rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* Info callout */
    .info-callout {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
DEFAULTS = {
    "uploaded_files":       {},   # {filename: bytes}
    "dataframes":           {},   # {filename: pd.DataFrame}
    "forecast_output":      None, # forecast result DataFrame → feeds Inventory
    "forecast_done":        False,
    "abc_xyz_data":         None,
    "active_module":        "Dashboard",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── File routing rules ─────────────────────────────────────────────────────────
FILE_ROUTES = {
    "motorin tüketim":  "forecast",
    "forecasteoq":      "inventory",
    # ABC-XYZ: any other xlsx/xls not matched above
}

def route_file(filename: str) -> str:
    """Determine which module a file belongs to by name pattern."""
    name_lower = filename.lower().replace(" ", " ")
    for pattern, module in FILE_ROUTES.items():
        if pattern in name_lower:
            return module
    return "abc_xyz"   # default for unrecognised files


def load_dataframe(filename: str, raw_bytes: bytes) -> pd.DataFrame | None:
    """Load Excel/CSV bytes into a DataFrame safely."""
    try:
        ext = Path(filename).suffix.lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(raw_bytes))
        elif ext == ".csv":
            return pd.read_csv(io.BytesIO(raw_bytes))
        else:
            return None          # docx → no DataFrame needed here
    except Exception as e:
        st.error(f"Could not parse **{filename}**: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Decision Support System")
    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────────────
    st.markdown("### Upload Data Files")
    uploaded = st.file_uploader(
        label="Upload files (xlsx, xls, docx)",
        type=["xlsx", "xls", "docx", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload all your data files here. They will be routed to the correct module automatically.",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state["uploaded_files"]:
                raw = f.read()
                st.session_state["uploaded_files"][f.name] = raw
                df = load_dataframe(f.name, raw)
                if df is not None:
                    st.session_state["dataframes"][f.name] = df
                    module_tag = route_file(f.name)
                    st.success(f"{f.name} → **{module_tag}**")

    # ── Uploaded files list ────────────────────────────────────────────────────
    if st.session_state["uploaded_files"]:
        st.markdown("**Loaded files:**")
        for fname in st.session_state["uploaded_files"]:
            tag = route_file(fname)
            color = {"forecast": "#3b82f6", "inventory": "#10b981",
                     "abc_xyz": "#f59e0b"}.get(tag, "#94a3b8")
            st.markdown(
                f"<small><span style='color:{color}'>&#9679;</span> {fname}</small>",
                unsafe_allow_html=True,
            )
        if st.button("Clear all files", type="secondary", use_container_width=True):
            for k in ["uploaded_files", "dataframes", "forecast_output",
                      "forecast_done", "abc_xyz_data"]:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()

    st.markdown("---")

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.markdown("### Navigation")
    pages = [
        "Dashboard",
        "EDA",
        "ABC-XYZ Analysis",
        "Forecast",
        "Inventory (EOQ)",
        "Time Series Analysis",
    ]
    icons = {
        "Dashboard":            "[ DSH ]",
        "EDA":                  "[ EDA ]",
        "ABC-XYZ Analysis":     "[ ABC ]",
        "Forecast":             "[ FOR ]",
        "Inventory (EOQ)":      "[ INV ]",
        "Time Series Analysis": "[ TSA ]",
    }

    selected = st.radio(
        "Go to",
        pages,
        index=pages.index(st.session_state["active_module"])
              if st.session_state["active_module"] in pages else 0,
        format_func=lambda x: f"{icons[x]}  {x}",
        label_visibility="collapsed",
    )
    st.session_state["active_module"] = selected

    # Forecast → Inventory handoff indicator
    if st.session_state["forecast_done"]:
        st.markdown(
            '<div class="info-callout">Forecast output is ready and will be used in <b>Inventory (EOQ)</b>.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("v1.0 | Decision Support System")


# ── Helper: get the right DataFrame for a module ──────────────────────────────
def get_df_for_module(module_tag: str) -> pd.DataFrame | None:
    """Return the first DataFrame whose filename routes to this module."""
    for fname, df in st.session_state["dataframes"].items():
        if route_file(fname) == module_tag:
            return df, fname
    return None, None


# ── Module: Dashboard ─────────────────────────────────────────────────────────
def render_dashboard():
    st.markdown('<div class="module-header">Summary Dashboard</div>', unsafe_allow_html=True)

    files_loaded   = len(st.session_state["uploaded_files"])
    forecast_ready = st.session_state["forecast_done"]
    dfs_loaded     = len(st.session_state["dataframes"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files Uploaded",   files_loaded)
    c2.metric("DataFrames Parsed", dfs_loaded)
    c3.metric("Forecast Ready",   "Yes" if forecast_ready else "No")
    c4.metric("Modules",          6)

    st.markdown("---")
    st.markdown("#### File Routing Map")

    if not st.session_state["uploaded_files"]:
        st.info("No files uploaded yet. Use the sidebar to upload your data files.")
        st.markdown("""
        **Expected files:**
        | File | Module |
        |---|---|
        | `motorin tüketim.xlsx` | Forecast |
        | `ForecastEOQ.xlsx` | Inventory (EOQ) |
        | Any other xlsx / xls | ABC-XYZ Analysis |
        | Any .docx | Reference / EDA |
        """)
    else:
        rows = []
        for fname in st.session_state["uploaded_files"]:
            has_df = fname in st.session_state["dataframes"]
            rows.append({
                "File Name":  fname,
                "Routed To":  route_file(fname),
                "DataFrame":  "Yes" if has_df else "N/A (docx)",
                "Shape":      str(st.session_state["dataframes"][fname].shape)
                              if has_df else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if forecast_ready and st.session_state["forecast_output"] is not None:
        st.markdown("---")
        st.markdown("#### Forecast Output Preview (fed into Inventory)")
        st.dataframe(
            st.session_state["forecast_output"].head(10),
            use_container_width=True,
        )


# ── Module: EDA ───────────────────────────────────────────────────────────────
def render_eda():
    st.markdown('<div class="module-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if not st.session_state["dataframes"]:
        st.warning("Please upload at least one Excel/CSV file to begin EDA.")
        return

    # File selector
    file_options = list(st.session_state["dataframes"].keys())
    chosen_file  = st.selectbox("Select file to analyse", file_options)
    df = st.session_state["dataframes"][chosen_file]

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistics", "Distributions", "Correlations"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))
        st.markdown("**First 20 rows**")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Column types**")
        dtypes_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Dtype":  df.dtypes.values.astype(str),
            "Nulls":  df.isna().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**Descriptive statistics (numeric)**")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    with tab3:
        st.markdown('<div class="info-callout">Connect your EDA visualisation code here (histograms, box plots, etc.)</div>',
                    unsafe_allow_html=True)
        # ── PLUG YOUR EDA VISUALISATION CODE HERE ──────────────────────────────
        # from modules.eda import run_distributions
        # run_distributions(df)
        # ───────────────────────────────────────────────────────────────────────
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Column to plot", numeric_cols)
            st.bar_chart(df[col].dropna().value_counts().head(30))

    with tab4:
        numeric_df = df.select_dtypes("number")
        if len(numeric_df.columns) >= 2:
            st.dataframe(numeric_df.corr().round(3), use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation.")


# ── Module: ABC-XYZ ───────────────────────────────────────────────────────────
def render_abc_xyz():
    st.markdown('<div class="module-header">ABC-XYZ Analysis</div>', unsafe_allow_html=True)

    df, fname = get_df_for_module("abc_xyz")

    if df is None:
        st.warning("No ABC-XYZ data file found. Upload an Excel file that is NOT named 'motorin tüketim' or 'ForecastEOQ'.")
        return

    st.info(f"Using file: **{fname}**  |  Shape: {df.shape}")

    # ── PLUG YOUR ABC-XYZ CODE HERE ────────────────────────────────────────────
    # from modules.abc_xyz import run_abc_xyz
    # result = run_abc_xyz(df)
    # st.dataframe(result, use_container_width=True)
    # ───────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="info-callout">Paste / import your ABC-XYZ analysis code in <code>modules/abc_xyz.py</code> and call it here.</div>',
                unsafe_allow_html=True)
    with st.expander("Raw data preview"):
        st.dataframe(df.head(50), use_container_width=True)


# ── Module: Forecast ──────────────────────────────────────────────────────────
def render_forecast():
    st.markdown('<div class="module-header">Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-callout">Forecast results are automatically stored in session state and forwarded to <b>Inventory (EOQ)</b>.</div>',
        unsafe_allow_html=True,
    )

    df, fname = get_df_for_module("forecast")

    if df is None:
        st.warning("Please upload **motorin tüketim.xlsx** to run the forecast.")
        return

    st.info(f"Using file: **{fname}**  |  Shape: {df.shape}")

    with st.expander("Raw data preview"):
        st.dataframe(df.head(30), use_container_width=True)

    if st.button("Run Forecast", type="primary", use_container_width=False):
        with st.spinner("Running forecast model..."):

            # ── PLUG YOUR FORECAST CODE HERE ───────────────────────────────────
            # from modules.forecast import run_forecast
            # forecast_df = run_forecast(df)
            # ───────────────────────────────────────────────────────────────────

            # PLACEHOLDER — replace with your actual forecast result:
            forecast_df = df.copy()   # <-- swap this line with your actual output
            forecast_df["_source"] = "forecast"

            # CRITICAL: store in session state so Inventory can read it
            st.session_state["forecast_output"] = forecast_df
            st.session_state["forecast_done"]   = True

        st.success("Forecast complete! Results saved and ready for Inventory (EOQ).")
        st.dataframe(forecast_df.head(20), use_container_width=True)

    elif st.session_state["forecast_done"]:
        st.info("Forecast already run. Results are available in Inventory (EOQ).")
        st.dataframe(
            st.session_state["forecast_output"].head(20),
            use_container_width=True,
        )


# ── Module: Inventory (EOQ) ───────────────────────────────────────────────────
def render_inventory():
    st.markdown('<div class="module-header">Inventory Optimisation (EOQ / ROP)</div>', unsafe_allow_html=True)

    # Priority: use forecast output; fallback to ForecastEOQ.xlsx
    if st.session_state["forecast_done"] and st.session_state["forecast_output"] is not None:
        df       = st.session_state["forecast_output"]
        src_note = "Using **forecast output** as demand input."
        badge    = "badge-ok"
    else:
        df, fname = get_df_for_module("inventory")
        if df is None:
            st.warning(
                "No inventory data found. Either:\n"
                "- Run the **Forecast** module first, OR\n"
                "- Upload **ForecastEOQ.xlsx**"
            )
            return
        src_note = f"Using fallback file: **{fname}** (run Forecast first for live demand data)."
        badge    = "badge-warn"

    st.markdown(f'<div class="info-callout">{src_note}</div>', unsafe_allow_html=True)

    with st.expander("Input data preview"):
        st.dataframe(df.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("#### EOQ Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        holding_cost    = st.number_input("Holding cost per unit / year (TL)", value=50.0, min_value=0.01)
    with col2:
        ordering_cost   = st.number_input("Ordering cost per order (TL)",      value=200.0, min_value=0.01)
    with col3:
        lead_time_days  = st.number_input("Lead time (days)",                   value=7,    min_value=1)

    if st.button("Calculate EOQ / ROP", type="primary"):
        with st.spinner("Calculating..."):

            # ── PLUG YOUR INVENTORY CODE HERE ──────────────────────────────────
            # from modules.inventory import run_inventory
            # result = run_inventory(df, holding_cost, ordering_cost, lead_time_days)
            # ───────────────────────────────────────────────────────────────────

            # PLACEHOLDER example (swap with your code):
            import math
            st.markdown("#### Results")
            # Example assumes df has a column with demand — adjust column name:
            numeric_cols = df.select_dtypes("number").columns.tolist()
            if numeric_cols:
                demand_col  = st.selectbox("Select demand column", numeric_cols)
                annual_D    = float(df[demand_col].sum())
                eoq         = math.sqrt((2 * annual_D * ordering_cost) / holding_cost)
                daily_d     = annual_D / 365
                rop         = daily_d * lead_time_days

                r1, r2, r3 = st.columns(3)
                r1.metric("Annual Demand",  f"{annual_D:,.0f}")
                r2.metric("EOQ",            f"{eoq:,.1f} units")
                r3.metric("Reorder Point",  f"{rop:,.1f} units")
            else:
                st.warning("No numeric columns found in the data.")


# ── Module: Time Series Analysis ──────────────────────────────────────────────
def render_time_series():
    st.markdown('<div class="module-header">Time Series Analysis</div>', unsafe_allow_html=True)

    if not st.session_state["dataframes"]:
        st.warning("Please upload a data file first.")
        return

    file_options = list(st.session_state["dataframes"].keys())
    chosen_file  = st.selectbox("Select file", file_options, key="ts_file")
    df           = st.session_state["dataframes"][chosen_file]

    st.info(f"File: **{chosen_file}**  |  Shape: {df.shape}")

    # ── PLUG YOUR TIME SERIES CODE HERE ────────────────────────────────────────
    # from modules.time_series import run_time_series
    # run_time_series(df)
    # ───────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="info-callout">Paste / import your Time Series Analysis code in <code>modules/time_series.py</code>.</div>',
                unsafe_allow_html=True)

    col_options = df.columns.tolist()
    date_col    = st.selectbox("Date / time column",  col_options, key="ts_date")
    value_col   = st.selectbox("Value column",        df.select_dtypes("number").columns.tolist(), key="ts_val")

    try:
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna().set_index(date_col).sort_index()
        st.line_chart(ts_df)
    except Exception as e:
        st.error(f"Could not plot time series: {e}")


# ── Router ─────────────────────────────────────────────────────────────────────
MODULE_RENDERERS = {
    "Dashboard":            render_dashboard,
    "EDA":                  render_eda,
    "ABC-XYZ Analysis":     render_abc_xyz,
    "Forecast":             render_forecast,
    "Inventory (EOQ)":      render_inventory,
    "Time Series Analysis": render_time_series,
}
MODULE_RENDERERS[st.session_state["active_module"]]()

MODULE_RENDERERS[st.session_state["active_module"]]()
