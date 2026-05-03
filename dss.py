import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def clean_raw(df):
    rename_map = {
        "Malzeme": "Material",
        "Malzeme kısa metni": "Description",
        "Hareket türleri metni": "MovementType",
        "Kayıt tarihi": "Date",
        "Miktar Abs": "Quantity",
        "Temel ölçü birimi": "Unit",
        "WhichDepo?": "Warehouse"
    }
    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Material"] = df["Material"].astype(str)
    df = df.dropna(subset=["Date"])
    return df

# =============================================================================
# 2. AGGREGATION FUNCTIONS
# =============================================================================

def get_monthly(df):
    result = {}
    for (code, desc), grp in df.groupby(["Material", "Description"]):
        m = grp.set_index("Date")["Quantity"].resample("MS").sum()
        if len(m) < 6:
            continue
        full = pd.date_range(start=m.index.min(), end=m.index.max(), freq="MS")
        m = m.reindex(full, fill_value=0)
        result[(code, desc)] = m
    return result

def get_weekly(df):
    result = {}
    for (code, desc), grp in df.groupby(["Material", "Description"]):
        w = grp.set_index("Date")["Quantity"].resample("W").sum()
        if len(w) < 20:
            continue
        full = pd.date_range(start=w.index.min(), end=w.index.max(), freq="W")
        w = w.reindex(full, fill_value=0)
        result[(code, desc)] = w
    return result

# =============================================================================
# 3. FORECAST MODELS
# =============================================================================

def arima_forecast(series, horizon=6):
    try:
        model = ARIMA(series, order=(1,1,1))
        res = model.fit()
        fc = res.forecast(steps=horizon)
        return fc
    except:
        return None

def make_features(series, lags=12):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series))
    return df.dropna()

def ml_forecast(series, model_type="xgboost"):
    df = make_features(series)
    if len(df) < 20:
        return None, None, None, None

    X = df.drop("y", axis=1)
    y = df["y"]

    split = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    else:
        model = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=4,
            random_seed=42, verbose=0
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return y_test, preds, mae, rmse

# =============================================================================
# 4. ABC – XYZ ANALYSIS
# =============================================================================
def color_abc_xyz(val):
    # ABC renkleri
    abc_colors = {
        "A": "#4CAF50",   # yeşil
        "B": "#FFC107",   # sarı
        "C": "#F44336"    # kırmızı
    }

    # XYZ renkleri
    xyz_colors = {
        "X": "#4CAF50",   # yeşil
        "Y": "#FFC107",   # sarı
        "Z": "#F44336"    # kırmızı
    }

    if val in abc_colors:
        return f"background-color: {abc_colors[val]}; color: white; font-weight: bold;"
    if val in xyz_colors:
        return f"background-color: {xyz_colors[val]}; color: white; font-weight: bold;"
    return ""
    
def abc_analysis(monthly_dict):
    records = []
    for (code, desc), series in monthly_dict.items():
        total = series.sum()
        records.append([code, desc, total])

    df = pd.DataFrame(records, columns=["Material", "Description", "TotalDemand"])
    df = df.sort_values("TotalDemand", ascending=False)
    df["Cumulative"] = df["TotalDemand"].cumsum()
    df["CumulativeRatio"] = df["Cumulative"] / df["TotalDemand"].sum()

    def abc_class(x):
        if x <= 0.80:
            return "A"
        elif x <= 0.95:
            return "B"
        else:
            return "C"

    df["ABC"] = df["CumulativeRatio"].apply(abc_class)
    return df

def xyz_analysis(monthly_dict):
    records = []
    for (code, desc), series in monthly_dict.items():
        mean = series.mean()
        std = series.std()
        cv = std / mean if mean > 0 else 999

        if cv < 0.5:
            xyz = "X"
        elif cv < 1.0:
            xyz = "Y"
        else:
            xyz = "Z"

        records.append([code, desc, mean, std, cv, xyz])

    df = pd.DataFrame(records, columns=["Material", "Description", "Mean", "Std", "CV", "XYZ"])
    return df

st.subheader("ABC–XYZ Matrix")

merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material")
merged["ABC_XYZ"] = merged["ABC"] + merged["XYZ"]

styled = merged.style.applymap(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])

st.dataframe(styled, use_container_width=True)

# =============================================================================
# 5. INVENTORY OPTIMIZATION
# =============================================================================

def inventory_metrics(series, service=0.95, lt=7, order_cost=2000, hold_pct=0.25, unit_cost=1):
    if len(series) < 6:
        return None

    mean_m = series.mean()
    std_m = series.std()

    mean_d = mean_m / 30
    std_d = std_m / np.sqrt(30)

    if mean_d <= 0:
        return None

    z = norm.ppf(service)
    mu_L = mean_d * lt
    sigma_L = std_d * np.sqrt(lt)

    ss = max(0, z * sigma_L)
    rop = mu_L + ss

    annual = mean_d * 365
    hold_cost = hold_pct * unit_cost
    eoq = np.sqrt((2 * annual * order_cost) / hold_cost)

    return {
        "mean_monthly": mean_m,
        "std_monthly": std_m,
        "mean_daily": mean_d,
        "std_daily": std_d,
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq": eoq
    }

# =============================================================================
# 6. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="DSS", layout="wide")
st.title("📦 Integrated Decision Support System")

# -------------------------
# STEP 1 — DATA UPLOAD
# -------------------------
st.header("1️⃣ Data Upload")
uploaded = st.file_uploader("Upload Excel (xlsx/xls) or CSV", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Please upload a file to continue.")
    st.stop()

df = load_file(uploaded)
df = clean_raw(df)

st.success("Data successfully loaded!")

# -------------------------
# STEP 2 — EDA
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 EDA", "⏳ Time Series", "🧮 ABC–XYZ", "📈 Forecast", "📦 Inventory", "📋 Dashboard", "ℹ Info"]
)

with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.write(df.head())
    st.write("Total records:", len(df))

# -------------------------
# STEP 3 — TIME SERIES
# -------------------------
monthly = get_monthly(df)
weekly = get_weekly(df)

items = [f"{c} - {d}" for (c,d) in monthly.keys()]
selected = st.sidebar.selectbox("Select a product", items)

key = list(monthly.keys())[items.index(selected)]
code, desc = key
series_m = monthly[key]

with tab2:
    st.header("⏳ Time Series")
    st.line_chart(series_m)

# -------------------------
# STEP 4 — ABC–XYZ
# -------------------------
with tab3:
    st.header("🧮 ABC – XYZ Analysis")

    df_abc = abc_analysis(monthly)
    df_xyz = xyz_analysis(monthly)

    st.subheader("ABC Analysis")
    st.dataframe(df_abc)

    st.subheader("XYZ Analysis")
    st.dataframe(df_xyz)

    st.subheader("ABC–XYZ Matrix")
    merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material")
    merged["ABC_XYZ"] = merged["ABC"] + merged["XYZ"]
    st.dataframe(merged)

# -------------------------
# STEP 5 — FORECAST
# -------------------------
with tab4:
    st.header("📈 Forecasting Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("ARIMA")
        fc_arima = arima_forecast(series_m)
        if fc_arima is not None:
            st.line_chart(fc_arima)
        else:
            st.warning("ARIMA forecast could not be generated.")

    with col2:
        st.subheader("XGBoost")
        y_test_xgb, preds_xgb, mae_xgb, rmse_xgb = ml_forecast(series_m, "xgboost")
        if preds_xgb is not None:
            st.write(f"MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
            st.line_chart(pd.DataFrame({"Actual": y_test_xgb, "Prediction": preds_xgb}))
        else:
            st.warning("Insufficient data.")

    with col3:
        st.subheader("CatBoost")
        y_test_cat, preds_cat, mae_cat, rmse_cat = ml_forecast(series_m, "catboost")
        if preds_cat is not None:
            st.write(f"MAE: {mae_cat:.2f}, RMSE: {rmse_cat:.2f}")
            st.line_chart(pd.DataFrame({"Actual": y_test_cat, "Prediction": preds_cat}))
        else:
            st.warning("Insufficient data.")

# -------------------------
# STEP 6 — INVENTORY OPTIMIZATION
# -------------------------
with tab5:
    st.header("📦 Inventory Optimization")

    service = st.slider("Service Level", 0.80, 0.999, 0.95)
    lt = st.number_input("Lead Time (days)", 1, 60, 7)
    order_cost = st.number_input("Order Cost", 1.0, 10000.0, 2000.0)
    hold_pct = st.number_input("Holding Cost Rate", 0.01, 1.0, 0.25)
    unit_cost = st.number_input("Unit Cost", 0.1, 1000.0, 1.0)

    metrics = inventory_metrics(series_m, service, lt, order_cost, hold_pct, unit_cost)

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Safety Stock", f"{metrics['safety_stock']:.1f}")
        col2.metric("Reorder Point (ROP)", f"{metrics['reorder_point']:.1f}")
        col3.metric("EOQ", f"{metrics['eoq']:.1f}")
    else:
        st.warning("Not enough data for inventory calculations.")

# -------------------------
# STEP 7 — DASHBOARD
# -------------------------
with tab6:
    st.header("📋 Dashboard — Summary")

    st.subheader("Product Information")
    st.write(f"**Code:** {code}")
    st.write(f"**Description:** {desc}")

    st.subheader("Monthly Demand")
    st.line_chart(series_m)

    st.subheader("Forecast Results")
    if fc_arima is not None:
        st.write("ARIMA Forecast")
        st.line_chart(fc_arima)

    if preds_xgb is not None:
        st.write("XGBoost Forecast")
        st.line_chart(pd.DataFrame({"Actual": y_test_xgb, "Prediction": preds_xgb}))

    if preds_cat is not None:
        st.write("CatBoost Forecast")
        st.line_chart(pd.DataFrame({"Actual": y_test_cat, "Prediction": preds_cat}))

    st.subheader("Inventory Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Avg", f"{metrics['mean_monthly']:.1f}")
    col2.metric("Monthly Std", f"{metrics['std_monthly']:.1f}")
    col3.metric("Daily Avg", f"{metrics['mean_daily']:.2f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Daily Std", f"{metrics['std_daily']:.2f}")
    col5.metric("Safety Stock", f"{metrics['safety_stock']:.1f}")
    col6.metric("ROP", f"{metrics['reorder_point']:.1f}")
    
    st.metric("EOQ", f"{metrics['eoq']:.1f}")
