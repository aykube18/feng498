import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import norm
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt

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

# =============================================================================
# 5. INVENTORY LOGIC (UPDATED, SIMULATION-BASED)
# =============================================================================

SERVICE_LEVEL = 0.95
ORDERING_COST = 2000.0
HOLDING_COST_PCT = 0.25
UNIT_COST = 1.0
SIMULATION_DAYS = 365
DEFAULT_LEAD_TIME_DAYS = 7

@dataclass
class MaterialItem:
    code: str
    name: str
    mean_daily: float
    std_daily: float
    distribution: str = "gamma"
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS

def calculate_reorder_point(mean_daily: float, std_daily: float,
                            lead_time_days: int,
                            service_level: float,
                            distribution: str = "normal") -> float:
    if mean_daily <= 0:
        return 0.0

    lt_mean = mean_daily * lead_time_days
    lt_std = std_daily * math.sqrt(lead_time_days)

    try:
        if distribution == "normal":
            z = norm.ppf(service_level)
            return max(0.0, lt_mean + z * lt_std)
        else:
            # basitleştirilmiş: diğer dağılımlar için de normal yaklaşım
            z = norm.ppf(service_level)
            return max(0.0, lt_mean + z * lt_std)
    except:
        return max(0.0, lt_mean)

def calculate_eoq(mean_daily: float,
                  ordering_cost: float = ORDERING_COST,
                  holding_cost_pct: float = HOLDING_COST_PCT,
                  unit_cost: float = UNIT_COST) -> float:
    if mean_daily <= 0:
        return 1.0

    annual_demand = mean_daily * 365
    holding_cost = holding_cost_pct * unit_cost

    if holding_cost <= 0 or annual_demand <= 0:
        return max(mean_daily * 30, 1)

    try:
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(1.0, eoq)
    except:
        return max(mean_daily * 30, 1)

def _generate_daily_demand(mean_daily: float, std_daily: float,
                           distribution: str, day: int) -> float:
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

def simulate_inventory_with_real_data(item: MaterialItem,
                                      reorder_point: float,
                                      order_qty: float,
                                      simulation_days: int = SIMULATION_DAYS):
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
        # receive orders
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
        "min_inventory": float(inv_arr.min()),
        "max_inventory": float(inv_arr.max()),
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

# =============================================================================
# 6. COLOR FUNCTION FOR ABC–XYZ MATRIX
# =============================================================================

def color_abc_xyz(val):
    abc_colors = {"A": "#4CAF50", "B": "#FFC107", "C": "#F44336"}
    xyz_colors = {"X": "#4CAF50", "Y": "#FFC107", "Z": "#F44336"}
    if val in abc_colors:
        return f"background-color: {abc_colors[val]}; color: white; font-weight: bold;"
    if val in xyz_colors:
        return f"background-color: {xyz_colors[val]}; color: white; font-weight: bold;"
    return ""

# =============================================================================
# 7. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="DSS", layout="wide")
st.title("📦 Integrated Decision Support System")

st.header("1️⃣ Data Upload")
uploaded = st.file_uploader("Upload Excel (xlsx/xls) or CSV", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Please upload a file to continue.")
    st.stop()

df = load_file(uploaded)
df = clean_raw(df)
st.success("Data successfully loaded!")

# Correct logical order
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 EDA", "🧮 ABC–XYZ", "📈 Forecast", "📦 Inventory", "⏳ Time Series", "📋 Dashboard", "ℹ Info"]
)

# -------------------------
# TAB 1 — EDA
# -------------------------
with tab1:
    st.header("📊 Exploratory Data Analysis")
    st.write(df.head())
    st.write("Total records:", len(df))

# -------------------------
# PREPARE SERIES
# -------------------------
monthly = get_monthly(df)
weekly = get_weekly(df)

items = [f"{c} - {d}" for (c,d) in monthly.keys()]
selected = st.sidebar.selectbox("Select a product", items)

key = list(monthly.keys())[items.index(selected)]
code, desc = key
series_m = monthly[key]

# -------------------------
# TAB 2 — ABC–XYZ
# -------------------------
with tab2:
    st.header("🧮 ABC–XYZ Analysis")

    df_abc = abc_analysis(monthly)
    df_xyz = xyz_analysis(monthly)

    st.subheader("ABC Analysis")
    st.dataframe(df_abc)

    st.subheader("XYZ Analysis")
    st.dataframe(df_xyz)

    st.subheader("ABC–XYZ Matrix")
    merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material")
    merged["ABC_XYZ"] = merged["ABC"] + merged["XYZ"]

    styled = merged.style.applymap(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])
    st.dataframe(styled, use_container_width=True)

# -------------------------
# TAB 3 — FORECAST
# -------------------------
with tab3:
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
# TAB 4 — INVENTORY (SPLIT LAYOUT)
# -------------------------
with tab4:
    st.header("📦 Inventory Optimization — Simulation Based")

    # derive daily stats from monthly series
    mean_monthly = series_m.mean()
    std_monthly = series_m.std()
    mean_daily = mean_monthly / 30 if mean_monthly > 0 else 0.0
    std_daily = std_monthly / 30 if std_monthly > 0 else 0.0

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.subheader("Parameters")

        service = st.slider("Service Level", 0.80, 0.999, SERVICE_LEVEL)
        lt = st.number_input("Lead Time (days)", 1, 60, DEFAULT_LEAD_TIME_DAYS)
        order_cost = st.number_input("Order Cost", 1.0, 10000.0, ORDERING_COST)
        hold_pct = st.number_input("Holding Cost Rate", 0.01, 1.0, HOLDING_COST_PCT)
        unit_cost = st.number_input("Unit Cost", 0.1, 1000.0, UNIT_COST)

        if mean_daily <= 0:
            st.warning("Not enough demand data to compute inventory metrics.")
        else:
            item = MaterialItem(
                code=str(code),
                name=str(desc),
                mean_daily=mean_daily,
                std_daily=std_daily,
                distribution="gamma",
                lead_time_days=int(lt),
            )

            rop = calculate_reorder_point(item.mean_daily, item.std_daily,
                                          item.lead_time_days, service, item.distribution)
            eoq = calculate_eoq(item.mean_daily, order_cost, hold_pct, unit_cost)

            st.subheader("Key Inventory Metrics")
            st.metric("Reorder Point (ROP)", f"{rop:.1f}")
            st.metric("EOQ", f"{eoq:.1f}")
            st.metric("Daily Mean Demand", f"{item.mean_daily:.2f}")
            st.metric("Daily Std Dev", f"{item.std_daily:.2f}")

            st.markdown("---")
            st.subheader("Simulation Results")

            result, history = simulate_inventory_with_real_data(item, rop, eoq)

            colm1, colm2 = st.columns(2)
            colm1.metric("Avg Inventory (≥0)", f"{result['avg_inventory']:.1f}")
            colm2.metric("Min Inventory", f"{result['min_inventory']:.1f}")
            colm3, colm4 = st.columns(2)
            colm3.metric("Max Inventory", f"{result['max_inventory']:.1f}")
            colm4.metric("Fill Rate", f"{result['fill_rate']:.1f}%")
            colm5, colm6 = st.columns(2)
            colm5.metric("Stockout Days", f"{result['stockout_days']}")
            colm6.metric("Total Shortage", f"{result['total_shortage']:.1f}")

    with col_right:
        if mean_daily <= 0:
            st.info("Inventory plots will appear here once demand statistics are available.")
        else:
            st.subheader("Inventory & Demand Over Time")

            days = history["days"]
            inventory = history["inventory"]
            demand = history["demand"]

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(days, inventory, label="Inventory Level", color="#DC2626", linewidth=1.8)
            ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax1.axhline(rop, color="#16A34A", linestyle="--", linewidth=1.5, label="ROP")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Inventory")
            ax1.set_title(f"Inventory Trajectory — {code}")
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(days, demand, color="#2563EB", linewidth=1.5)
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Demand")
            ax2.set_title("Daily Demand (Simulated)")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.hist(demand, bins=30, color="#0EA5E9", alpha=0.8, edgecolor="white")
            ax3.set_xlabel("Demand")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Demand Distribution (Simulated)")
            st.pyplot(fig3)

# -------------------------
# TAB 5 — TIME SERIES
# -------------------------
with tab5:
    st.header("⏳ Time Series")
    st.line_chart(series_m)

# -------------------------
# TAB 6 — DASHBOARD
# -------------------------
with tab6:
    st.header("📋 Dashboard — Summary")

    st.subheader("Product Information")
    st.write(f"**Code:** {code}")
    st.write(f"**Description:** {desc}")

    st.subheader("Monthly Demand")
    st.line_chart(series_m)

    st.subheader("Forecast Results")
    if 'fc_arima' in locals() and fc_arima is not None:
        st.write("ARIMA Forecast")
        st.line_chart(fc_arima)

    if 'preds_xgb' in locals() and preds_xgb is not None:
        st.write("XGBoost Forecast")
        st.line_chart(pd.DataFrame({"Actual": y_test_xgb, "Prediction": preds_xgb}))

    if 'preds_cat' in locals() and preds_cat is not None:
        st.write("CatBoost Forecast")
        st.line_chart(pd.DataFrame({"Actual": y_test_cat, "Prediction": preds_cat}))

    if mean_daily > 0:
        st.subheader("Inventory Metrics Snapshot")
        st.metric("Daily Mean Demand", f"{mean_daily:.2f}")
        st.metric("Daily Std Dev", f"{std_daily:.2f}")
        st.metric("Service Level", f"{SERVICE_LEVEL*100:.0f}%")

# -------------------------
# TAB 7 — INFO
# -------------------------
with tab7:
    st.header("ℹ Info")
    st.markdown("""
This integrated DSS includes:
- EDA
- ABC–XYZ classification
- Forecasting (ARIMA, XGBoost, CatBoost)
- Simulation-based inventory optimization (ROP, EOQ, stockouts, fill rate)
- Time series visualization
- Summary dashboard
""")
