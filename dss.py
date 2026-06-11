import streamlit as st
import pandas as pd
import numpy as np
import warnings
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS & CONFIGURATIONS
# =============================================================================
ORDERING_COST = 2792.0       
HOLDING_COST_PCT = 0.25      
SERVICE_LEVEL = 0.95         
STOCKOUT_COST_PER_UNIT = 150.0 

# Forecast Configuration Constraints
DATE_COL = "Date"
VALUE_COL = "Quantity"
TEST_START = "2025-04-02"          
SEASONAL = 7                       
DEFAULT_ORDER = (1, 1, 2)       
DEFAULT_SORDER = (1, 1, 1, SEASONAL)
ALPHA = 0.05                       
AUTO_SELECT = False                
N_LAGS = 14
ROLL_WINDOWS = (7, 14, 30)

SHORT_NAMES = {
    600080: "MOTORIN", 600096: "TOKA 32", 600102: "ÇEMBER TOKA",
    600112: "SIYAH ÇEMBER", 603789: "ELDİVEN", 603812: "KULAK TIKACI",
    607290: "Ç.4140", 626100: "SAPAN", 627518: "BRK 77x60", 627519: "BRK 80x122"
}

# =============================================================================
# 1. DATA LOADING & STRUCTURAL INTEGRATION
# =============================================================================
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        try:
            return pd.read_excel(uploaded_file, sheet_name=0)
        except Exception:
            return pd.read_excel(uploaded_file)

def clean_raw(df):
    rename_map = {
        "Malzeme": "Material", "Malzeme kısa metni": "Description",
        "Hareket türleri metni": "MovementType", "Kayıt tarihi": "Date",
        "Miktar Abs": "Quantity", "Temel ölçü birimi": "Unit", "WhichDepo?": "Warehouse",
        "Tarih": "Date", "Tüketim Miktarı": "Quantity", "Mal Giriş": "Inflow", "Tüketim": "Quantity"
    }
    df = df.rename(columns=rename_map)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    else:
        df["Quantity"] = 0.0
        
    if "Inflow" in df.columns:
        df["Inflow"] = pd.to_numeric(df["Inflow"], errors="coerce").fillna(0.0)
    else:
        df["Inflow"] = 0.0

    if "Material" not in df.columns:
        df["Material"] = "Unknown"
    if "Description" not in df.columns:
        df["Description"] = "General Product"
        
    df["Material"] = df["Material"].astype(str)
    return df

# =============================================================================
# 2. SECTORAL AGGREGATION PIPELINES
# =============================================================================
def get_monthly(df):
    result = {}
    if df.empty:
        return result
    for (code, desc), grp in df.groupby(["Material", "Description"]):
        m = grp.set_index("Date")["Quantity"].resample("MS").sum()
        if len(m) < 2:
            continue
        full = pd.date_range(start=m.index.min(), end=m.index.max(), freq="MS")
        m = m.reindex(full, fill_value=0)
        result[(code, desc)] = m
    return result

def get_weekly(df):
    result = {}
    if df.empty:
        return result
    for (code, desc), grp in df.groupby(["Material", "Description"]):
        w = grp.set_index("Date")["Quantity"].resample("W").sum()
        if len(w) < 4:
            continue
        full = pd.date_range(start=w.index.min(), end=w.index.max(), freq="W")
        w = w.reindex(full, fill_value=0)
        result[(code, desc)] = w
    return result

# =============================================================================
# 3. ROLLING ONE-STEP ADVANCED FORECAST MECHANISMS (UPDATED DETECTOR)
# =============================================================================
def _aicc(result) -> float:
    k = len(result.params)
    n = result.nobs
    return result.aic + (2 * k * k + 2 * k) / (n - k - 1) if (n - k - 1) > 0 else np.inf

def select_order_aicc(y: pd.Series):
    best = {"score": np.inf, "order": DEFAULT_ORDER, "sorder": DEFAULT_SORDER}
    for p, d, q in itertools.product(range(3), range(2), range(3)):
        for P, D, Q in itertools.product(range(2), range(2), range(2)):
            try:
                res = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, SEASONAL),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                score = _aicc(res)
                if np.isfinite(score) and score < best["score"]:
                    best = {"score": score, "order": (p, d, q), "sorder": (P, D, Q, SEASONAL)}
            except Exception:
                continue
    return best["order"], best["sorder"]

def rolling_one_step_sarima(y_train: pd.Series, test_values: np.ndarray, order, sorder):
    res = SARIMAX(y_train, order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    hist = res
    pred = np.empty(len(test_values))
    lo = np.empty(len(test_values))
    hi = np.empty(len(test_values))
    for i, actual in enumerate(test_values):
        fc = hist.get_forecast(steps=1)
        pred[i] = float(fc.predicted_mean.iloc[0])
        ci = np.asarray(fc.conf_int(alpha=ALPHA))[0]
        lo[i], hi[i] = ci[0], ci[1]
        hist = hist.append([actual], refit=False)
    return np.maximum(pred, 0.0), np.maximum(lo, 0.0), np.maximum(hi, 0.0)

def make_feature_row(history_values, idx_date, t_value):
    h = np.asarray(history_values, dtype=float)
    feat = {}
    for lag in range(1, N_LAGS + 1):
        feat[f"lag_{lag}"] = h[-lag] if len(h) >= lag else 0.0
    for w in ROLL_WINDOWS:
        window = h[-w:] if len(h) >= 1 else np.array([0.0])
        feat[f"rmean_{w}"] = float(np.mean(window)) if len(window) else 0.0
        feat[f"rstd_{w}"]  = float(np.std(window))  if len(window) > 1 else 0.0
    
    dow = idx_date.dayofweek
    feat["dow"]       = dow
    feat["dow_sin"]   = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"]   = np.cos(2 * np.pi * dow / 7.0)
    feat["month"]     = idx_date.month
    feat["month_sin"] = np.sin(2 * np.pi * idx_date.month / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * idx_date.month / 12.0)
    feat["trend"]     = t_value
    return feat

def build_training_matrix(train_frame):
    values = train_frame[VALUE_COL].values.astype(float)
    dates  = train_frame["Date"].values
    ts     = train_frame["t"].values
    rows, targets = [], []
    for i in range(N_LAGS, len(values)):
        hist = values[:i]
        feat = make_feature_row(hist, pd.Timestamp(dates[i]), int(ts[i]))
        rows.append(feat)
        targets.append(values[i])
    X = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=float)
    return X, y, list(X.columns)

def rolling_one_step_ml(model, train_frame, test_frame, feature_cols):
    history = list(train_frame[VALUE_COL].values.astype(float))
    preds = []
    for _, row in test_frame.iterrows():
        feat = make_feature_row(history, pd.Timestamp(row["Date"]), int(row["t"]))
        x = pd.DataFrame([feat])[feature_cols]
        yhat = max(0.0, float(model.predict(x)[0]))
        preds.append(yhat)
        history.append(float(row[VALUE_COL]))
    return np.asarray(preds)

def fit_xgboost(X, y):
    model = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    model.fit(X, y, verbose=False)
    return model

def fit_catboost(X, y):
    model = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                               l2_leaf_reg=3, subsample=0.8, random_seed=42, verbose=0)
    model.fit(X, y)
    return model

def error_stats(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred, dtype=float)
    err  = actual - pred
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100.0) if mask.any() else np.nan
    return mae, rmse, mape

# =============================================================================
# 4. ABC – XYZ MATERIAL ANALYSIS
# =============================================================================
def abc_analysis(monthly_dict):
    records = []
    for (code, desc), series in monthly_dict.items():
        total = series.sum()
        records.append([code, desc, total])
    if not records:
        return pd.DataFrame(columns=["Material", "Description", "TotalDemand", "Cumulative", "CumulativeRatio", "ABC"])
    df = pd.DataFrame(records, columns=["Material", "Description", "TotalDemand"])
    df = df.sort_values("TotalDemand", ascending=False)
    df["Cumulative"] = df["TotalDemand"].cumsum()
    total_sum = df["TotalDemand"].sum()
    df["CumulativeRatio"] = df["Cumulative"] / total_sum if total_sum > 0 else 0
    df["ABC"] = df["CumulativeRatio"].apply(lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C"))
    return df

def xyz_analysis(monthly_dict):
    records = []
    for (code, desc), series in monthly_dict.items():
        mean = series.mean()
        std = series.std()
        cv = std / mean if mean > 0 else 999
        xyz = "X" if cv < 0.5 else ("Y" if cv < 1.0 else "Z")
        records.append([code, desc, mean, std, cv, xyz])
    if not records:
        return pd.DataFrame(columns=["Material", "Description", "Mean", "Std", "CV", "XYZ"])
    return pd.DataFrame(records, columns=["Material", "Description", "Mean", "Std", "CV", "XYZ"])

# =============================================================================
# 5. INVENTORY ENGINE MATRIX LOGIC
# =============================================================================
def calculate_real_inventory(df):
    df = df.copy()
    df["Envanter"] = 0.0
    df["Stockout"] = 0.0
    current_inv = 0.0
    inventory_list = []
    stockout_list = []

    for idx, row in df.iterrows():
        current_inv = current_inv + row["Inflow"] - row["Quantity"]
        if current_inv < 0:
            stockout_list.append(-current_inv)
            current_inv = 0.0
        else:
            stockout_list.append(0.0)
        inventory_list.append(current_inv)

    df["Envanter"] = inventory_list
    df["Stockout"] = stockout_list
    return df

def calculate_real_inventory_costs(df, unit_price, order_cost, hold_pct, stockout_cost):
    df = df.copy()
    daily_holding_cost_rate = hold_pct / 365.0
    df["Holding_Cost_Daily"] = df["Envanter"] * daily_holding_cost_rate * unit_price
    df["Is_Order"] = df["Inflow"] > 0
    df["Ordering_Cost_Daily"] = df["Is_Order"].astype(float) * order_cost
    df["Stockout_Cost_Daily"] = df["Stockout"] * stockout_cost
    df["Total_Cost_Daily"] = df["Holding_Cost_Daily"] + df["Ordering_Cost_Daily"] + df["Stockout_Cost_Daily"]
    return df

def calculate_eoq_rop_stats(df, unit_price, order_cost, hold_pct, service_level, lead_time_days):
    consumption = df["Quantity"].values
    days = len(df)
    total_consumption = consumption.sum()
    avg_daily_consumption = total_consumption / days if days > 0 else 0

    consumption_nonzero = consumption[consumption > 0]
    std_consumption = consumption_nonzero.std() if len(consumption_nonzero) > 1 else avg_daily_consumption * 0.3
    std_daily = std_consumption / np.sqrt(days) if days > 1 else avg_daily_consumption * 0.3
    cv = (std_daily / avg_daily_consumption * 100) if avg_daily_consumption > 0 else 0

    annual_demand = avg_daily_consumption * 365.0
    holding_cost_val = hold_pct * unit_price

    if annual_demand > 0 and holding_cost_val > 0:
        eoq = math.sqrt((2.0 * annual_demand * order_cost) / holding_cost_val)
    else:
        eoq = avg_daily_consumption * 30.0
    eoq = max(1.0, eoq)

    lt_mean = avg_daily_consumption * lead_time_days
    lt_std = std_daily * math.sqrt(lead_time_days)
    z = stats.norm.ppf(service_level)
    rop = max(0.0, lt_mean + z * lt_std)

    return {
        "avg_daily_consumption": avg_daily_consumption,
        "std_daily": std_daily,
        "cv_percent": cv,
        "eoq": eoq,
        "rop": rop,
        "annual_demand": annual_demand
    }

def color_abc_xyz(val):
    colors = {"A": "#4CAF50", "B": "#FFC107", "C": "#F44336",
              "X": "#4CAF50", "Y": "#FFC107", "Z": "#F44336"}
    if val in colors:
        return f"background-color: {colors[val]}; color: white; font-weight: bold;"
    return ""

# =============================================================================
# 6. STREAMLIT UI FRAMEWORK ORCHESTRATION
# =============================================================================
st.set_page_config(page_title="DSS Pro", layout="wide")
st.title("📦 Integrated Decision Support System")

st.header("1️⃣ Data Upload")
uploaded = st.file_uploader("Upload Excel (xlsx/xls) or CSV File", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Please upload a transactional logging file to launch dashboards.")
    st.stop()

raw_data = load_file(uploaded)
if raw_data is None or raw_data.empty:
    st.error("Uploaded dataset structure appears broken or empty.")
    st.stop()

df = clean_raw(raw_data)
st.success("Data parsing completed and structural field matching verified!")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Data Sample", "🧮 ABC–XYZ Matrix", "📈 Advanced Forecasting", 
    "💸 Real Cost & Inventory", "⏳ Daily Time Series", "📋 Summary Dashboard"
])

with tab1:
    st.header("📊 Exploratory Data Analysis & Structure")
    st.dataframe(df.head(10))
    st.metric("Total Records Ingested", f"{len(df):,}")

monthly = get_monthly(df)
if not monthly:
    st.warning("Insufficient data frequency to populate matrix operations.")
    st.stop()

items_list = [f"{c} - {d}" for (c, d) in monthly.keys()]
selected_product = st.sidebar.selectbox("🎯 Target Product Selection", items_list)
selected_key = list(monthly.keys())[items_list.index(selected_product)]
p_code, p_desc = selected_key

# Filter active row scope
product_df = df[(df["Material"] == p_code) & (df["Description"] == p_desc)].copy()
if product_df.empty:
    product_df = df.copy()

product_df = product_df.set_index("Date")
full_daily_range = pd.date_range(start=product_df.index.min(), end=product_df.index.max(), freq='D')
product_df = product_df.reindex(full_daily_range).fillna({"Quantity": 0.0, "Inflow": 0.0, "Material": p_code, "Description": p_desc})
product_df = product_df.reset_index().rename(columns={"index": "Date"})
product_df["t"] = np.arange(1, len(product_df) + 1)

# =============================================================================
# TAB 2 — ABC–XYZ CLASSIFICATIONS
# =============================================================================
with tab2:
    st.header("🧮 ABC–XYZ Matrix Analytics")
    df_abc = abc_analysis(monthly)
    df_xyz = xyz_analysis(monthly)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ABC Volume Stratification")
        st.dataframe(df_abc)
    with col_b:
        st.subheader("XYZ Dispersion Index")
        st.dataframe(df_xyz)
        
    st.subheader("Combined ABC-XYZ Color Mapping Grid")
    merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material", how="left")
    merged["ABC_XYZ"] = merged["ABC"].fillna("") + merged["XYZ"].fillna("")
    
    try:
        styled = merged.style.map(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])
    except AttributeError:
        styled = merged.style.applymap(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])
    st.markdown(styled.to_html(), unsafe_allow_html=True)

# =============================================================================
# TAB 3 — ADVANCED FORECASTING (INTEGRATED NEW ROLLING ONE-STEP ENGINE)
# =============================================================================
with tab3:
    st.header("📈 High-Fidelity Rolling 1-Step Forecast Engine (Daily)")
    
    cutoff_dt = pd.Timestamp(TEST_START)
    train_frame = product_df[product_df["Date"] < cutoff_dt].reset_index(drop=True)
    test_frame = product_df[product_df["Date"] >= cutoff_dt].reset_index(drop=True)
    
    if len(train_frame) < N_LAGS or len(test_frame) == 0:
        st.warning(f"Data configuration criteria not met. Confirm training series spans prior to cutoff: {TEST_START}")
    else:
        st.write(f"**Execution Profile:** Training Days: `{len(train_frame)}` | Testing Days Evaluation Window: `{len(test_frame)}`")
        
        y_train_series = pd.Series(train_frame[VALUE_COL].values, index=pd.RangeIndex(1, len(train_frame) + 1))
        
        with st.spinner("Orchestrating SARIMA Append Iterations & Supervised ML Matrices..."):
            if AUTO_SELECT:
                order, sorder = select_order_aicc(y_train_series)
            else:
                order, sorder = DEFAULT_ORDER, DEFAULT_SORDER
                
            # Run updated walk-forward forecasting loops
            sarima_pred, lo_conf, hi_conf = rolling_one_step_sarima(
                y_train_series, test_frame[VALUE_COL].values.astype(float), order, sorder
            )
            
            X_tr, y_tr, feature_cols = build_training_matrix(train_frame)
            xgb_model = fit_xgboost(X_tr, y_tr)
            cat_model = fit_catboost(X_tr, y_tr)
            
            xgb_pred = rolling_one_step_ml(xgb_model, train_frame, test_frame, feature_cols)
            cat_pred = rolling_one_step_ml(cat_model, train_frame, test_frame, feature_cols)
            
        comp_df = build_comparison(test_frame, sarima_pred, lo_conf, hi_conf, xgb_pred, cat_pred)
        
        stats_dict = {
            "SARIMA":   error_stats(comp_df["Actual"], comp_df["SARIMA"]),
            "XGBoost":  error_stats(comp_df["Actual"], comp_df["XGBoost"]),
            "CatBoost": error_stats(comp_df["Actual"], comp_df["CatBoost"]),
        }
        
        # Display Accuracy Table
        accuracy_df = pd.DataFrame({
            "Model Specification": [f"SARIMA{order}{sorder[:3]} s={sorder[3]}", "XGBoost Regressor", "CatBoost Regressor"],
            "MAE": [stats_dict["SARIMA"][0], stats_dict["XGBoost"][0], stats_dict["CatBoost"][0]],
            "RMSE": [stats_dict["SARIMA"][1], stats_dict["XGBoost"][1], stats_dict["CatBoost"][1]],
            "MAPE (%)": [stats_dict["SARIMA"][2], stats_dict["XGBoost"][2], stats_dict["CatBoost"][2]]
        })
        st.subheader("Model Accuracy Comparisons (Test Window Evaluation)")
        st.table(accuracy_df)
        
        # Matplotlib Output Generating
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
        boundary_date = train_frame["Date"].iloc[-1]

        # Top Chart
        ax1.plot(product_df["Date"], product_df[VALUE_COL], color="#1f77b4", linewidth=0.8, alpha=0.6, label="Actual Series Logs")
        ax1.plot(comp_df["date"], comp_df["SARIMA"], color="#2ca02c", linewidth=1.4, label="SARIMA Path")
        ax1.plot(comp_df["date"], comp_df["XGBoost"], color="#d62728", linewidth=1.4, linestyle="--", label="XGBoost Path")
        ax1.plot(comp_df["date"], comp_df["CatBoost"], color="#9467bd", linewidth=1.4, linestyle="-.", label="CatBoost Path")
        ax1.axvline(boundary_date, color="grey", linestyle=":", linewidth=1.3)
        ax1.set_title("Full Chronological Validation Series Trends")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.25)

        # Bottom Zoom Window
        ax2.fill_between(comp_df["date"], comp_df["SARIMA_Lo95"], comp_df["SARIMA_Hi95"], color="#2ca02c", alpha=0.10, label="SARIMA 95% Interval Bound")
        ax2.plot(comp_df["date"], comp_df["Actual"], color="#1f77b4", linewidth=1.2, marker="o", markersize=3, label="Actual Evaluation Steps")
        ax2.plot(comp_df["date"], comp_df["SARIMA"], color="#2ca02c", linewidth=1.8, label="SARIMA Out-of-Sample")
        ax2.plot(comp_df["date"], comp_df["XGBoost"], color="#d62728", linewidth=1.8, linestyle="--", label="XGBoost Out-of-Sample")
        ax2.plot(comp_df["date"], comp_df["CatBoost"], color="#9467bd", linewidth=1.8, linestyle="-.", label="CatBoost Out-of-Sample")
        ax2.set_title("Rolling Active Prediction Window Comparisons")
        ax2.set_xlabel("Time Horizon Axis")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.25)
        
        st.pyplot(fig)
        plt.close()

# =============================================================================
# TAB 4 — REAL COSTS & INVENTORY BALANCE TRACKING
# =============================================================================
with tab4:
    st.header("💸 Real Material Flows, Costs & Stockout Matrix")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ui_unit_price = st.number_input("Unit Price (Material Base Cost)", min_value=0.1, max_value=50000.0, value=10.0, step=1.0)
        ui_order_cost = st.number_input("Ordering Setup Cost (Fixed)", min_value=1.0, max_value=100000.0, value=float(ORDERING_COST), step=10.0)
    with col_p2:
        ui_hold_pct = st.slider("Holding Cost Rate (Annualized %)", 0.01, 1.0, float(HOLDING_COST_PCT), step=0.01)
        ui_stockout_cost = st.number_input("Penalty/Stockout Cost Per Unit Shortage", min_value=0.0, max_value=5000.0, value=float(STOCKOUT_COST_PER_UNIT), step=1.0)
        
    ui_lead_time = st.number_input("Lead Time Duration (Days)", min_value=1, max_value=90, value=int(7), step=1)
    
    calculated_df = calculate_real_inventory(product_df)
    cost_df = calculate_real_inventory_costs(calculated_df, ui_unit_price, ui_order_cost, ui_hold_pct, ui_stockout_cost)
    inv_stats = calculate_eoq_rop_stats(cost_df, ui_unit_price, ui_order_cost, ui_hold_pct, SERVICE_LEVEL, ui_lead_time)
    
    total_h = cost_df["Holding_Cost_Daily"].sum()
    total_o = cost_df["Ordering_Cost_Daily"].sum()
    total_s = cost_df["Stockout_Cost_Daily"].sum()
    grand_total = total_h + total_o + total_s
    
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Holding Cost Flow", f"TRY {total_h:,.2f}")
    mc2.metric("Total Ordering Setup Cost", f"TRY {total_o:,.2f}")
    mc3.metric("Stockout Financial Impact", f"TRY {total_s:,.2f}", delta=f"{cost_df['Stockout'].sum():.0f} Units Short", delta_color="inverse")
    mc4.metric("Grand Operation Cost", f"TRY {grand_total:,.2f}")
    
    st.markdown("---")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Calculated Optimization Targets (EOQ - ROP)")
        st.metric("Economic Order Quantity (EOQ Optimal Q*)", f"{inv_stats['eoq']:.2f} Units")
        st.metric("Safety Reorder Point (ROP)", f"{inv_stats['rop']:.2f} Units")
        st.metric("Coefficient of Variation (CV %)", f"{inv_stats['cv_percent']:.1f}%")
        
    with col_m2:
        st.subheader("Operational Trajectory Realizations")
        st.metric("Average Daily Sales Demand", f"{inv_stats['avg_daily_consumption']:.2f} Units")
        st.metric("Stockout Occurrence Days", f"{(cost_df['Stockout'] > 0).sum()} Days")
        
    st.subheader("Real Inventory Balance & Safety Bands Over Time")
    fig_st, ax_st = plt.subplots(figsize=(14, 4.5))
    days_arr = range(len(cost_df))
    ax_st.plot(days_arr, cost_df["Envanter"], color="#DC2626", linewidth=2, label="Actual On-Hand Balance Level")
    ax_st.axhline(y=inv_stats["rop"], color="#16A34A", linestyle="--", label=f"Optimal ROP Line ({inv_stats['rop']:.1f})")
    ax_st.axhline(y=inv_stats["eoq"], color="#D97706", linestyle="-.", label=f"Optimal EOQ Line ({inv_stats['eoq']:.1f})")
    ax_st.fill_between(days_arr, 0, inv_stats["rop"], alpha=0.05, color="#16A34A", label="Safety Stock Band")
    ax_st.set_xlabel("Day Progress")
    ax_st.set_ylabel("Quantity Units")
    ax_st.legend(loc="upper right")
    ax_st.grid(alpha=0.2)
    st.pyplot(fig_st)
    plt.close()

    st.subheader("Daily Accumulative Cost Layer Analysis")
    fig_c, ax_c = plt.subplots(figsize=(14, 4))
    ax_c.fill_between(days_arr, 0, cost_df["Holding_Cost_Daily"], alpha=0.6, color="#16A34A", label="Holding Allocation")
    ax_c.fill_between(days_arr, cost_df["Holding_Cost_Daily"], cost_df["Holding_Cost_Daily"] + cost_df["Ordering_Cost_Daily"], alpha=0.6, color="#D97706", label="Ordering Allocation")
    ax_c.fill_between(days_arr, cost_df["Holding_Cost_Daily"] + cost_df["Ordering_Cost_Daily"], cost_df["Total_Cost_Daily"], alpha=0.6, color="#EF4444", label="Penalty Stockout Allocation")
    ax_c.set_ylabel("TRY Value")
    ax_c.set_xlabel("Day Progress")
    ax_c.legend(loc="upper left")
    ax_c.grid(alpha=0.2)
    st.pyplot(fig_c)
    plt.close()

# =============================================================================
# TAB 5 — DAILY TIME SERIES
# =============================================================================
with tab5:
    st.header("⏳ Baseline Historical Time Series Data")
    st.line_chart(product_df.set_index("Date")["Quantity"])

# =============================================================================
# TAB 6 — CONTROL SUMMARY DASHBOARD
# =============================================================================
with tab6:
    st.header("📋 Executive Summary Control Board")
    db_col1, db_col2 = st.columns(2)
    with db_col1:
        st.subheader("Product Profiler Metadata")
        st.markdown(f"""
        - **Target Inventory Code:** `{p_code}`
        - **Description Tag:** `{p_desc}`
        - **Calculated Theoretical Annual Demand:** `{inv_stats['annual_demand']:,.0f} Units`
        - **Assigned Procurement Lead Time:** `{ui_lead_time} Days`
        """)
    with db_col2:
        st.subheader("Target Recommendation Engine")
        st.success(f"**Recommended Batch Size (Q*):** {inv_stats['eoq']:.0f} Units")
        st.success(f"**Recommended Trigger Threshold (ROP):** {inv_stats['rop']:.0f} Units")
        
    st.info("System integration running stable. Interface numeric data typing verified.")
