import streamlit as st
import pandas as pd
import numpy as np
import warnings
import math
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS & CONFIGURATIONS
# =============================================================================
SERVICE_LEVEL = 0.95
ORDERING_COST = 2792.0       
HOLDING_COST_PCT = 0.25      
STOCKOUT_COST_PER_UNIT = 150.0 
SIMULATION_DAYS = 365
DEFAULT_LEAD_TIME_DAYS = 7

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

@dataclass
class MaterialItem:
    code: str
    name: str
    mean_daily: float
    std_daily: float
    distribution: str = "gamma"
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS

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
# 3. ROLLING ONE-STEP ADVANCED FORECAST MECHANISMS
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

def build_comparison(test, sarima_pred, lo, hi, xgb_pred, cat_pred) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": test["Date"].values,
        "t": test["t"].values,
        "Actual": test[VALUE_COL].values.astype(float),
        "SARIMA": sarima_pred,
        "SARIMA_Lo95": lo,
        "SARIMA_Hi95": hi,
        "XGBoost": xgb_pred,
        "CatBoost": cat_pred,
    })
    return out

# =============================================================================
# 4. ABC – XYZ ANALYSIS
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
# 5. INVENTORY LOGIC (SIMULATION BASED & FORECAST-DRIVEN)
# =============================================================================
def calculate_reorder_point(mean_daily: float, std_daily: float, lead_time_days: int, service_level: float) -> float:
    if mean_daily <= 0:
        return 0.0
    lt_mean = mean_daily * lead_time_days
    lt_std = std_daily * math.sqrt(lead_time_days)
    try:
        z = norm.ppf(service_level)
        return max(0.0, lt_mean + z * lt_std)
    except Exception:
        return max(0.0, lt_mean)

def calculate_eoq(mean_daily: float, ordering_cost: float, holding_cost_pct: float, unit_cost: float) -> float:
    if mean_daily <= 0:
        return 1.0
    annual_demand = mean_daily * 365.0
    holding_cost = holding_cost_pct * unit_cost
    if holding_cost <= 0 or annual_demand <= 0:
        return max(mean_daily * 30.0, 1.0)
    try:
        eoq = math.sqrt((2.0 * annual_demand * ordering_cost) / holding_cost)
        return max(1.0, eoq)
    except Exception:
        return max(mean_daily * 30.0, 1.0)

def _generate_daily_demand(mean_daily: float, std_daily: float, distribution: str, day: int) -> float:
    day_factor = 0.8 if day % 7 in [5, 6] else 1.0
    spike = np.random.gamma(2, 10) if np.random.random() < 0.05 else 0.0

    if distribution == "normal":
        base = max(0.0, np.random.normal(mean_daily, std_daily))
    elif distribution == "gamma":
        if std_daily <= 0 or mean_daily <= 0:
            base = mean_daily
        else:
            shape = (mean_daily ** 2) / (std_daily ** 2)
            scale = (std_daily ** 2) / mean_daily
            base = np.random.gamma(shape=shape, scale=scale)
    elif distribution == "lognormal":
        if std_daily <= 0 or mean_daily <= 0:
            base = mean_daily
        else:
            sigma = np.sqrt(np.log((std_daily ** 2 / mean_daily ** 2) + 1))
            mu = np.log(mean_daily) - (sigma ** 2) / 2
            base = np.random.lognormal(mean=mu, sigma=sigma)
    else:
        base = mean_daily

    return max(0.0, base * day_factor + spike)

def simulate_inventory_with_real_data(item: MaterialItem, reorder_point: float, order_qty: float, simulation_days: int = SIMULATION_DAYS):
    daily_demands = [
        _generate_daily_demand(item.mean_daily, item.std_daily, item.distribution, day)
        for day in range(simulation_days)
    ]

    inventory_history = []
    demand_history = []
    order_history = []
    pending_orders = []

    current_inventory = order_qty
    total_demand = 0.0
    total_shortage = 0.0
    total_fulfilled = 0.0
    stockout_days = 0

    for day in range(simulation_days):
        received = 0
        still_pending = []
        for arrival_day, qty in pending_orders:
            if day >= arrival_day:
                received += qty
            else:
                still_pending.append((arrival_day, qty))
        pending_orders = still_pending
        current_inventory += received

        demand = daily_demands[day]
        total_demand += demand
        demand_history.append(demand)

        if current_inventory >= demand:
            fulfilled = demand
            current_inventory -= fulfilled
            total_fulfilled += fulfilled
        else:
            fulfilled = current_inventory
            shortage = demand - fulfilled
            current_inventory = -shortage
            total_shortage += shortage
            stockout_days += 1
            total_fulfilled += fulfilled

        if current_inventory <= reorder_point and len(pending_orders) == 0:
            arrival_day = day + item.lead_time_days
            pending_orders.append((arrival_day, order_qty))
            order_history.append((day, order_qty))

        inventory_history.append(current_inventory)

    inv_arr = np.array(inventory_history, dtype=float)
    fill_rate = (total_fulfilled / total_demand * 100) if total_demand > 0 else 0.0

    result = {
        "avg_inventory": float(inv_arr[inv_arr >= 0].mean()) if np.any(inv_arr >= 0) else 0.0,
        "min_inventory": float(inv_arr.min()) if len(inv_arr) > 0 else 0.0,
        "max_inventory": float(inv_arr.max()) if len(inv_arr) > 0 else 0.0,
        "total_demand": total_demand,
        "total_shortage": total_shortage,
        "stockout_days": stockout_days,
        "fill_rate": fill_rate,
        "reorder_point": reorder_point,
        "order_quantity": order_qty,
    }

    history = {
        "days": list(range(simulation_days)),
        "inventory": inventory_history,
        "demand": demand_history,
        "orders": order_history,
    }

    return result, history

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

# PRE-CALCULATE FORECAST PIPELINE BEFORE TABS FOR INTER-TAB DEPENDENCY
cutoff_dt = pd.Timestamp(TEST_START)
monthly = get_monthly(df)

if not monthly:
    st.warning("Insufficient data frequency to populate analytics matrix operations.")
    st.stop()

items_list = [f"{c} - {d}" for (c, d) in monthly.keys()]
selected_product = st.sidebar.selectbox("🎯 Target Product Selection", items_list)
selected_key = list(monthly.keys())[items_list.index(selected_product)]
p_code, p_desc = selected_key

product_df = df[(df["Material"] == p_code) & (df["Description"] == p_desc)].copy()
if product_df.empty:
    product_df = df.copy()

product_df = product_df.set_index("Date")
full_daily_range = pd.date_range(start=product_df.index.min(), end=product_df.index.max(), freq='D')
product_df = product_df.reindex(full_daily_range).fillna({"Quantity": 0.0, "Inflow": 0.0, "Material": p_code, "Description": p_desc})
product_df = product_df.reset_index().rename(columns={"index": "Date"})
product_df["t"] = np.arange(1, len(product_df) + 1)

train_frame = product_df[product_df["Date"] < cutoff_dt].reset_index(drop=True)
test_frame = product_df[product_df["Date"] >= cutoff_dt].reset_index(drop=True)

# CORE ARDIŞIK ENTEGRASYON: Tahmin modelini sekmelerden önce çalıştırıp en başarılısını buluyoruz
has_valid_forecast = False
if len(train_frame) >= N_LAGS and len(test_frame) > 0:
    y_train_series = pd.Series(train_frame[VALUE_COL].values, index=pd.RangeIndex(1, len(train_frame) + 1))
    
    if AUTO_SELECT:
        order, sorder = select_order_aicc(y_train_series)
    else:
        order, sorder = DEFAULT_ORDER, DEFAULT_SORDER
        
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
    
    # MAE'ye göre en başarılı modeli seçiyoruz
    maes = {"SARIMA": stats_dict["SARIMA"][0], "XGBoost": stats_dict["XGBoost"][0], "CatBoost": stats_dict["CatBoost"][0]}
    best_model_name = min(maes, key=maes.get)
    best_forecast_series = comp_df[best_model_name]
    has_valid_forecast = True

# UI Tabs Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Data Sample", "🧮 ABC–XYZ Matrix", "📈 Advanced Forecasting", 
    "📦 Inventory Simulation", "⏳ Daily Time Series", "📋 Summary Dashboard"
])

# -------------------------
# TAB 1 — EDA
# -------------------------
with tab1:
    st.header("📊 Exploratory Data Analysis & Structure")
    st.dataframe(df.head(10))
    st.metric("Total Records Ingested", f"{len(df):,}")

# -------------------------
# TAB 2 — ABC–XYZ
# -------------------------
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
# TAB 3 — ADVANCED FORECASTING
# =============================================================================
with tab3:
    st.header("📈 High-Fidelity Rolling 1-Step Forecast Engine (Daily)")
    if not has_valid_forecast:
        st.warning(f"Data configuration criteria not met for forecast loops.")
    else:
        st.write(f"**Execution Profile:** Training Days: `{len(train_frame)}` | Testing Days Evaluation Window: `{len(test_frame)}`")
        st.info(f"🏆 **Dinamik Model Seçici:** En düşük MAE değerine sahip **{best_model_name}** modeli otomatik olarak kazandı ve Envanter simülasyonunu beslemek üzere atandı.")
        
        accuracy_df = pd.DataFrame({
            "Model Specification": [f"SARIMA{order}{sorder[:3]} s={sorder[3]}", "XGBoost Regressor", "CatBoost Regressor"],
            "MAE": [stats_dict["SARIMA"][0], stats_dict["XGBoost"][0], stats_dict["CatBoost"][0]],
            "RMSE": [stats_dict["SARIMA"][1], stats_dict["XGBoost"][1], stats_dict["CatBoost"][1]],
            "MAPE (%)": [stats_dict["SARIMA"][2], stats_dict["XGBoost"][2], stats_dict["CatBoost"][2]]
        })
        st.table(accuracy_df)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
        boundary_date = train_frame["Date"].iloc[-1]

        ax1.plot(product_df["Date"], product_df[VALUE_COL], color="#1f77b4", linewidth=0.8, alpha=0.6, label="Actual Series Logs")
        ax1.plot(comp_df["date"], comp_df["SARIMA"], color="#2ca02c", linewidth=1.4, label="SARIMA Path")
        ax1.plot(comp_df["date"], comp_df["XGBoost"], color="#d62728", linewidth=1.4, linestyle="--", label="XGBoost Path")
        ax1.plot(comp_df["date"], comp_df["CatBoost"], color="#9467bd", linewidth=1.4, linestyle="-.", label="CatBoost Path")
        ax1.axvline(boundary_date, color="grey", linestyle=":", linewidth=1.3)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.25)

        ax2.fill_between(comp_df["date"], comp_df["SARIMA_Lo95"], comp_df["SARIMA_Hi95"], color="#2ca02c", alpha=0.10, label="SARIMA 95% Interval Bound")
        ax2.plot(comp_df["date"], comp_df["Actual"], color="#1f77b4", linewidth=1.2, marker="o", markersize=3, label="Actual Evaluation Steps")
        ax2.plot(comp_df["date"], comp_df["SARIMA"], color="#2ca02c", linewidth=1.8, label="SARIMA Out-of-Sample")
        ax2.plot(comp_df["date"], comp_df["XGBoost"], color="#d62728", linewidth=1.8, linestyle="--", label="XGBoost Out-of-Sample")
        ax2.plot(comp_df["date"], comp_df["CatBoost"], color="#9467bd", linewidth=1.8, linestyle="-.", label="CatBoost Out-of-Sample")
        ax2.set_xlabel("Time Horizon Axis")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close()

# =============================================================================
# TAB 4 — INVENTORY OPTIMIZATION (TAHMİN BESLEMELİ SİMÜLASYON)
# =============================================================================
with tab4:
    st.header("📦 Inventory Optimization — Forecast-Driven Simulation")

    # BAĞIMLI DEĞİŞKENLER: Parametre hesaplamaları en iyi modelin tahmin serisinden türetilir
    if has_valid_forecast:
        mean_daily = float(best_forecast_series.mean())
        std_daily = float(best_forecast_series.std())
        st.caption(f"💡 **Veri Bağlantısı Aktif:** Envanter optimizasyon parametreleri, geleceğe yönelik en başarılı tahmin modelinden (**{best_model_name}**) dinamik olarak beslenmektedir.")
    else:
        mean_monthly = monthly[selected_key].mean()
        std_monthly = monthly[selected_key].std()
        mean_daily = mean_monthly / 30.0 if mean_monthly > 0 else 0.0
        std_daily = std_monthly / 30.0 if std_monthly > 0 else 0.0
        st.caption("⚠️ **Veri Bağlantısı Yedek Modda:** Yetersiz tahmin verisi sebebiyle geçmiş ham veri ortalamaları kullanılmaktadır.")

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.subheader("Simulation Parameters")
        service = st.slider("Service Level", 0.80, 0.999, float(SERVICE_LEVEL))
        lt = st.number_input("Lead Time (days)", 1, 60, int(DEFAULT_LEAD_TIME_DAYS))
        order_cost = st.number_input("Order Cost", 1.0, 10000.0, float(ORDERING_COST))
        hold_pct = st.number_input("Holding Cost Rate", 0.01, 1.0, float(HOLDING_COST_PCT))
        unit_cost = st.number_input("Unit Cost", 0.1, 1000.0, float(10.0)) # Cast to float

        if mean_daily <= 0:
            st.warning("Not enough demand data to compute inventory metrics.")
            result = None
            history = None
        else:
            item = MaterialItem(
                code=str(p_code),
                name=str(p_desc),
                mean_daily=mean_daily,
                std_daily=std_daily,
                distribution="gamma",
                lead_time_days=int(lt),
            )

            rop = calculate_reorder_point(item.mean_daily, item.std_daily, item.lead_time_days, service)
            eoq = calculate_eoq(item.mean_daily, order_cost, hold_pct, unit_cost)

            st.subheader("Key Forecast-Based Metrics")
            st.metric("Reorder Point (ROP Target)", f"{rop:.1f} Units")
            st.metric("Economic Order Quantity (EOQ Target Q*)", f"{eoq:.1f} Units")
            st.metric("Predicted Daily Mean Demand", f"{item.mean_daily:.2f}")
            st.metric("Predicted Daily Std Dev", f"{item.std_daily:.2f}")

            st.markdown("---")
            st.subheader("Simulation Operational Run Results")
            result, history = simulate_inventory_with_real_data(item, rop, eoq)

            colm1, colm2 = st.columns(2)
            colm1.metric("Avg Inventory (>=0)", f"{result['avg_inventory']:.1f}")
            colm2.metric("Min Inventory", f"{result['min_inventory']:.1f}")
            colm3, colm4 = st.columns(2)
            colm3.metric("Max Inventory", f"{result['max_inventory']:.1f}")
            colm4.metric("Fill Rate", f"{result['fill_rate']:.1f}%")
            colm5, colm6 = st.columns(2)
            colm5.metric("Stockout Days Triggers", f"{result['stockout_days']} Days")
            colm6.metric("Total Shortage Vol", f"{result['total_shortage']:.1f} Units")

    with col_right:
        if mean_daily <= 0 or result is None or history is None:
            st.info("Inventory plots will appear here once simulation calculation completes successfully.")
        else:
            st.subheader("Simulated Dynamic Inventory & Predicted Demand Over Time")

            days_axis = history["days"]
            inv_level = history["inventory"]
            sim_demand = history["demand"]

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(days_axis, inv_level, label="Inventory Level State", color="#DC2626", linewidth=1.8)
            ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax1.axhline(result["reorder_point"], color="#16A34A", linestyle="--", linewidth=1.5, label=f"ROP ({result['reorder_point']:.1f})")
            ax1.set_xlabel("Day Progress")
            ax1.set_ylabel("Inventory Balance")
            ax1.set_title(f"Dynamic Saw-Tooth Inventory Trajectory - Material: {p_code}")
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(days_axis, sim_demand, color="#2563EB", linewidth=1.5)
            ax2.set_xlabel("Day Progress")
            ax2.set_ylabel("Simulated Short Horizon Demand")
            ax2.set_title("Forecast-Driven Daily Demand Pattern (Simulated Execution)")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.hist(sim_demand, bins=30, color="#0EA5E9", alpha=0.8, edgecolor="white")
            ax3.set_xlabel("Demand Size Category")
            ax3.set_ylabel("Frequency Hit Metrics")
            ax3.set_title("Forecast Density Spread (Simulated Frequency Distribution)")
            st.pyplot(fig3)
            plt.close(fig3)

# -------------------------
# TAB 5 — TIME SERIES
# -------------------------
with tab5:
    st.header("⏳ Baseline Historical Time Series Data")
    st.line_chart(product_df.set_index("Date")["Quantity"])

# =============================================================================
# TAB 6 — SUMMARY CONTROL BOARD
# =============================================================================
with tab6:
    st.header("📋 Executive Summary Control Board")

    if not monthly:
        st.info("Dashboard summary blocks are currently locked.")
    else:
        st.subheader("Product Profiler Metadata")
        st.write(f"**Target Material Code:** `{p_code}`")
        st.write(f"**Description Tag:** `{p_desc}`")

        st.subheader("Historical Demand Baseline Overview")
        st.line_chart(product_df.set_index("Date")["Quantity"])

        if has_valid_forecast:
            st.subheader("Deployable Optimization Recommendations")
            st.success(f"🎯 **Best-Performing Model Asset Identified:** `{best_model_name}`")
            
            sc1, sc2 = st.columns(2)
            sc1.metric("Recommended Batch Size (Optimal Q*)", f"{eoq:.0f} Units")
            sc2.metric("Recommended Trigger Point (Optimal ROP)", f"{rop:.0f} Units")
            
            st.metric("Forecast-Based Daily Average Sales Demand", f"{mean_daily:.2f} Units/Day")
            st.metric("Calculated Baseline Service Level Policy Target", f"{service * 100:.1f}%")
