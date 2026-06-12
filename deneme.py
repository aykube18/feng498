# ENTEGRASYON NOTU
# Eski UI korunmuştur.
# Yeni envanter modeli dosyası ayrıca eklenmiştir.
# Bu çıktı manuel entegrasyon incelemesi için oluşturulmuştur.

########################
# ESKI UI KODU
########################

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
# ============================================================================

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
    if df.empty:
        return result
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
    if df.empty:
        return result
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
    if series is None or len(series) < 6:
        return None
    try:
        model = ARIMA(series, order=(1,1,1))
        res = model.fit()
        fc = res.forecast(steps=horizon)
        return fc
    except Exception:
        return None

def make_features(series, lags=12):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series))
    return df.dropna()

def ml_forecast(series, model_type="xgboost"):
    if series is None or len(series) < 20:
        return None, None, None, None

    df = make_features(series)
    if len(df) < 20:
        return None, None, None, None

    X = df.drop("y", axis=1)
    y = df["y"]

    split = int(len(df)*0.8)
    if split == 0 or split >= len(df):
        return None, None, None, None

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

    if not records:
        return pd.DataFrame(columns=["Material", "Description", "TotalDemand", "Cumulative", "CumulativeRatio", "ABC"])

    df = pd.DataFrame(records, columns=["Material", "Description", "TotalDemand"])
    df = df.sort_values("TotalDemand", ascending=False)
    df["Cumulative"] = df["TotalDemand"].cumsum()
    total_sum = df["TotalDemand"].sum()
    if total_sum == 0:
        df["CumulativeRatio"] = 0
    else:
        df["CumulativeRatio"] = df["Cumulative"] / total_sum

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

    if not records:
        return pd.DataFrame(columns=["Material", "Description", "Mean", "Std", "CV", "XYZ"])

    df = pd.DataFrame(records, columns=["Material", "Description", "Mean", "Std", "CV", "XYZ"])
    return df

# =============================================================================
# 5. INVENTORY LOGIC (SIMULATION-BASED)
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
        z = norm.ppf(service_level)
        return max(0.0, lt_mean + z * lt_std)
    except Exception:
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
    except Exception:
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
if df is None or df.empty:
    st.error("Loaded file is empty or invalid.")
    st.stop()

df = clean_raw(df)
st.success("Data successfully loaded!")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 EDA", "🧮 ABC–XYZ", "📈 Forecast", "📦 Inventory", "⏳ Time Series", "📋 Summary Dashboard"]
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

if not monthly:
    st.warning("No sufficient monthly data for time series and ABC-XYZ. Check your file.")
    series_m = None
    code, desc = "", ""
else:
    items_list = [f"{c} - {d}" for (c, d) in monthly.keys()]
    selected = st.sidebar.selectbox("Select a product", items_list)
    key = list(monthly.keys())[items_list.index(selected)]
    code, desc = key
    series_m = monthly[key]

# -------------------------
# TAB 2 — ABC–XYZ
# -------------------------
with tab2:
    st.header("🧮 ABC–XYZ Analysis")

    if not monthly:
        st.info("ABC-XYZ analysis is not available because monthly data is insufficient.")
    else:
        df_abc = abc_analysis(monthly)
        df_xyz = xyz_analysis(monthly)

        st.subheader("ABC Analysis")
        st.dataframe(df_abc)

        st.subheader("XYZ Analysis")
        st.dataframe(df_xyz)

        st.subheader("ABC-XYZ Matrix")
        try:
            merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material", how="left")
            merged["ABC_XYZ"] = merged["ABC"].fillna("") + merged["XYZ"].fillna("")

            if isinstance(merged, pd.DataFrame) and not merged.empty:
                # -------------------------------------------------------
                # DUZELTME: applymap() pandas>=2.1'de kaldirildi.
                # Yeni API: Styler.map() kullanimiyla degistirildi.
                # -------------------------------------------------------
                try:
                    # pandas >= 2.1: map() kullan
                    styled = merged.style.map(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])
                except AttributeError:
                    # pandas < 2.1: applymap() kullan (eski surum fallback)
                    styled = merged.style.applymap(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])

                st.markdown(styled.to_html(), unsafe_allow_html=True)
            else:
                st.warning("ABC-XYZ matrix could not be created. Check data consistency.")
        except Exception as e:
            st.error(f"Error while creating ABC-XYZ matrix: {e}")

# -------------------------
# TAB 3 — FORECAST
# -------------------------
with tab3:
    st.header("📈 Forecasting Models")

    if series_m is None:
        st.info("Forecasting is not available because no product could be selected.")
    else:
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
                st.warning("Insufficient data for XGBoost.")

        with col3:
            st.subheader("CatBoost")
            y_test_cat, preds_cat, mae_cat, rmse_cat = ml_forecast(series_m, "catboost")
            if preds_cat is not None:
                st.write(f"MAE: {mae_cat:.2f}, RMSE: {rmse_cat:.2f}")
                st.line_chart(pd.DataFrame({"Actual": y_test_cat, "Prediction": preds_cat}))
            else:
                st.warning("Insufficient data for CatBoost.")

# -------------------------
# TAB 4 — INVENTORY (SPLIT LAYOUT)
# -------------------------
with tab4:
    st.header("📦 Inventory Optimization — Simulation Based")

    if series_m is None:
        st.info("Inventory optimization is not available because no product could be selected.")
    else:
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
                result = None
                history = None
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
                colm1.metric("Avg Inventory (>=0)", f"{result['avg_inventory']:.1f}")
                colm2.metric("Min Inventory", f"{result['min_inventory']:.1f}")
                colm3, colm4 = st.columns(2)
                colm3.metric("Max Inventory", f"{result['max_inventory']:.1f}")
                colm4.metric("Fill Rate", f"{result['fill_rate']:.1f}%")
                colm5, colm6 = st.columns(2)
                colm5.metric("Stockout Days", f"{result['stockout_days']}")
                colm6.metric("Total Shortage", f"{result['total_shortage']:.1f}")

        with col_right:
            if series_m is None or mean_daily <= 0 or result is None or history is None:
                st.info("Inventory plots will appear here once demand statistics and simulation are available.")
            else:
                st.subheader("Inventory & Demand Over Time")

                days = history["days"]
                inventory = history["inventory"]
                demand = history["demand"]

                fig1, ax1 = plt.subplots(figsize=(8, 4))
                ax1.plot(days, inventory, label="Inventory Level", color="#DC2626", linewidth=1.8)
                ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
                ax1.axhline(result["reorder_point"], color="#16A34A", linestyle="--",
                            linewidth=1.5, label="ROP")
                ax1.set_xlabel("Day")
                ax1.set_ylabel("Inventory")
                ax1.set_title(f"Inventory Trajectory - {code}")
                ax1.legend()
                ax1.grid(alpha=0.3)
                st.pyplot(fig1)
                plt.close(fig1)

                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(days, demand, color="#2563EB", linewidth=1.5)
                ax2.set_xlabel("Day")
                ax2.set_ylabel("Demand")
                ax2.set_title("Daily Demand (Simulated)")
                ax2.grid(alpha=0.3)
                st.pyplot(fig2)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(figsize=(8, 3))
                ax3.hist(demand, bins=30, color="#0EA5E9", alpha=0.8, edgecolor="white")
                ax3.set_xlabel("Demand")
                ax3.set_ylabel("Frequency")
                ax3.set_title("Demand Distribution (Simulated)")
                st.pyplot(fig3)
                plt.close(fig3)

# -------------------------
# TAB 5 — TIME SERIES
# -------------------------
with tab5:
    st.header("⏳ Time Series")
    if series_m is None:
        st.info("Time series is not available because no product could be selected.")
    else:
        st.line_chart(series_m)

# -------------------------
# TAB 6 — DASHBOARD
# -------------------------
with tab6:
    st.header("📋 Summary Dashboard")

    if series_m is None:
        st.info("Dashboard is not available because no product could be selected.")
    else:
        st.subheader("Product Information")
        st.write(f"**Code:** {code}")
        st.write(f"**Description:** {desc}")

        st.subheader("Monthly Demand")
        st.line_chart(series_m)

        st.subheader("Forecast Results")
        if "fc_arima" in locals() and fc_arima is not None:
            st.write("ARIMA Forecast")
            st.line_chart(fc_arima)

        if "preds_xgb" in locals() and preds_xgb is not None:
            st.write("XGBoost Forecast")
            st.line_chart(pd.DataFrame({"Actual": y_test_xgb, "Prediction": preds_xgb}))

        if "preds_cat" in locals() and preds_cat is not None:
            st.write("CatBoost Forecast")
            st.line_chart(pd.DataFrame({"Actual": y_test_cat, "Prediction": preds_cat}))

        mean_monthly = series_m.mean()
        std_monthly = series_m.std()
        mean_daily = mean_monthly / 30 if mean_monthly > 0 else 0.0
        std_daily = std_monthly / 30 if std_monthly > 0 else 0.0

        st.subheader("Inventory Metrics Snapshot")
        st.metric("Daily Mean Demand", f"{mean_daily:.2f}")
        st.metric("Daily Std Dev", f"{std_daily:.2f}")
        st.metric("Default Service Level", f"{SERVICE_LEVEL*100:.0f}%")

########################
# YENI ENVANTER MODELI
########################

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from typing import Dict, List, Tuple
import math
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION FOR ForecastEOQ.xlsx
# ══════════════════════════════════════════════════════════════════════════════

EXCEL_FILE = "/content/ForecastEOQ.xlsx"
OUTPUT_DIR = "/content/"
MATERIAL_CODES = [600080, 600096, 600102, 600112, 603789, 603812, 607290, 626100, 627518, 627519]

SHORT_NAMES: Dict[int, str] = {
    600080: "MOTORIN",
    600096: "TOKA 32",
    600102: "ÇEMBER TOKA",
    600112: "SIYAH ÇEMBER",
    603789: "ELDİVEN",
    603812: "KULAK TIKACI",
    607290: "Ç.4140",
    626100: "SAPAN",
    627518: "BRK 77x60",
    627519: "BRK 80x122",
}

# ══════════════════════════════════════════════════════════════════════════════
# ORDERING COST PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

ORDERING_COST = 2792  # TRY - Fixed ordering cost per order
HOLDING_COST_PCT = 0.25  # 25% - Annual holding cost percentage
SERVICE_LEVEL = 0.95  # 95% service level
STOCKOUT_COST_PER_UNIT = 500  # TRY - Cost per unit of stockout (lost sales + penalty)

COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#6366F1",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA FROM EXCEL - ForecastEOQ.xlsx (REAL DATA)
# ══════════════════════════════════════════════════════════════════════════════

def load_material_data(filepath: str, material_code: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Load real data for a material from ForecastEOQ.xlsx.

    Columns: Tarih, Mal Giriş, Tüketim Miktarı, Unit Price, Para Birimi, Lead time
    """
    try:
        df = pd.read_excel(filepath, sheet_name=str(material_code))

        # Clean column names
        df.columns = df.columns.str.strip()

        # Convert date column to datetime
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df = df.sort_values("Tarih").reset_index(drop=True)

        # Fill NaN values with 0
        df["Mal Giriş"] = df["Mal Giriş"].fillna(0)
        df["Tüketim Miktarı"] = df["Tüketim Miktarı"].fillna(0)

        # Extract metadata (from first row)
        unit_price = float(df["Unit Price"].iloc[0]) if "Unit Price" in df.columns else 1.0
        currency = str(df["Para Birimi"].iloc[0]) if "Para Birimi" in df.columns else "TRY"
        lead_time = int(df["Lead time"].iloc[0]) if "Lead time" in df.columns else 2

        metadata = {
            "unit_price": unit_price,
            "currency": currency,
            "lead_time": lead_time,
            "material_code": material_code,
            "material_name": SHORT_NAMES.get(material_code, str(material_code)),
        }

        return df, metadata

    except Exception as e:
        print(f"❌ Error reading material code {material_code} → {str(e)}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE REAL INVENTORY (Allow negative for stockout visibility)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_real_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real Inventory = Previous Inventory + Inflow - Consumption
    
    Allows negative inventory (stockout) to show below zero line.
    """
    df = df.copy()
    df["Envanter"] = 0.0
    df["Stockout"] = 0.0

    current_inv = 0.0
    inventory_list = []
    stockout_list = []

    for idx, row in df.iterrows():
        current_inv = current_inv + row["Mal Giriş"] - row["Tüketim Miktarı"]
        
        if current_inv < 0:
            stockout_list.append(-current_inv)  # Record shortage
            # Keep negative value for visualization
        else:
            stockout_list.append(0)
        
        inventory_list.append(current_inv)

    df["Envanter"] = inventory_list
    df["Stockout"] = stockout_list
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE REAL INVENTORY COSTS (including stockout costs)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_real_inventory_costs(df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate real inventory costs based on actual inflow and consumption data:
    - Holding Cost = Daily Inventory × Daily Holding Cost Rate
    - Ordering Cost = Number of Orders × Ordering Cost per Order
    - Stockout Cost = Shortage Quantity × Stockout Cost per Unit
    """
    
    df = df.copy()
    unit_price = metadata["unit_price"]
    
    # Daily holding cost rate (annual rate / 365)
    daily_holding_cost_rate = HOLDING_COST_PCT / 365
    
    # Daily holding cost = Inventory × Daily Rate × Unit Price (only for positive inventory)
    df["Holding_Cost_Daily"] = df["Envanter"].clip(lower=0) * daily_holding_cost_rate * unit_price
    
    # Detect orders (when inflow > 0)
    df["Is_Order"] = df["Mal Giriş"] > 0
    df["Ordering_Cost_Daily"] = df["Is_Order"] * ORDERING_COST
    
    # Stockout cost = Shortage Quantity × Cost per Unit
    df["Stockout_Cost_Daily"] = df["Stockout"] * STOCKOUT_COST_PER_UNIT
    
    # Total daily cost
    df["Total_Cost_Daily"] = df["Holding_Cost_Daily"] + df["Ordering_Cost_Daily"] + df["Stockout_Cost_Daily"]
    
    # Calculate summary statistics
    total_holding_cost = df["Holding_Cost_Daily"].sum()
    total_ordering_cost = df["Ordering_Cost_Daily"].sum()
    total_stockout_cost = df["Stockout_Cost_Daily"].sum()
    total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost
    
    days = len(df)
    avg_daily_cost = total_cost / days if days > 0 else 0
    annual_cost = avg_daily_cost * 365
    
    total_orders = df["Is_Order"].sum()
    total_stockout_units = df["Stockout"].sum()
    stockout_days = (df["Stockout"] > 0).sum()
    
    cost_summary = {
        "material_code": metadata["material_code"],
        "material_name": metadata["material_name"],
        "days_analyzed": days,
        "total_holding_cost": round(total_holding_cost, 2),
        "total_ordering_cost": round(total_ordering_cost, 2),
        "total_stockout_cost": round(total_stockout_cost, 2),
        "total_cost": round(total_cost, 2),
        "avg_daily_cost": round(avg_daily_cost, 2),
        "annual_cost": round(annual_cost, 2),
        "total_orders": int(total_orders),
        "avg_order_qty": round(df[df["Mal Giriş"] > 0]["Mal Giriş"].mean(), 2) if total_orders > 0 else 0,
        "total_stockout_units": round(total_stockout_units, 2),
        "stockout_days": int(stockout_days),
        "unit_price": metadata["unit_price"],
        "currency": metadata["currency"],
    }
    
    return df, cost_summary


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_statistics(df: pd.DataFrame, metadata: Dict) -> Dict:
    """
    Calculate EOQ and ROP based on real data.
    """
    consumption = df["Tüketim Miktarı"].values
    inflow = df["Mal Giriş"].values
    inventory = df["Envanter"].values

    # Daily averages
    days = len(df)
    total_consumption = consumption.sum()
    total_inflow = inflow.sum()
    avg_daily_consumption = total_consumption / days if days > 0 else 0

    # Standard deviation (from Consumption)
    consumption_nonzero = consumption[consumption > 0]
    std_consumption = consumption_nonzero.std() if len(consumption_nonzero) > 1 else avg_daily_consumption * 0.3

    # Daily standard deviation
    std_daily = std_consumption / np.sqrt(len(df)) if len(df) > 1 else avg_daily_consumption * 0.3

    # Coefficient of Variation
    cv = (std_daily / avg_daily_consumption * 100) if avg_daily_consumption > 0 else 0

    # EOQ Calculation
    annual_demand = avg_daily_consumption * 365
    holding_cost = HOLDING_COST_PCT * metadata["unit_price"]

    if annual_demand > 0 and holding_cost > 0:
        eoq = math.sqrt((2 * annual_demand * ORDERING_COST) / holding_cost)
    else:
        eoq = avg_daily_consumption * 30

    eoq = max(1.0, eoq)

    # ROP Calculation (stock required during Lead Time)
    lead_time = metadata["lead_time"]
    lt_mean = avg_daily_consumption * lead_time
    lt_std = std_daily * math.sqrt(lead_time)

    # Z-score (for 95% service level)
    z = stats.norm.ppf(SERVICE_LEVEL)
    rop = max(0.0, lt_mean + z * lt_std)

    return {
        "material_code": metadata["material_code"],
        "material_name": metadata["material_name"],
        "total_days": days,
        "total_consumption": total_consumption,
        "total_inflow": total_inflow,
        "avg_daily_consumption": round(avg_daily_consumption, 2),
        "std_daily": round(std_daily, 2),
        "cv_percent": round(cv, 1),
        "max_inventory": float(inventory.max()),
        "min_inventory": float(inventory.min()),
        "avg_inventory": float(inventory.mean()),
        "eoq": round(eoq, 2),
        "rop": round(rop, 2),
        "unit_price": metadata["unit_price"],
        "currency": metadata["currency"],
        "lead_time": lead_time,
        "annual_demand": round(annual_demand, 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT INVENTORY COST DATA TO EXCEL/CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_inventory_cost_data(df: pd.DataFrame, cost_summary: Dict, output_dir: str) -> None:
    """
    Export detailed inventory cost data to Excel & CSV (including stockout)
    """
    
    export_df = pd.DataFrame({
        "Tarih (Date)": df["Tarih"],
        "Mal Giriş (Inflow)": df["Mal Giriş"],
        "Tüketim Miktarı (Consumption)": df["Tüketim Miktarı"],
        "Envanter (Inventory)": df["Envanter"],
        "Stockout (Units)": df["Stockout"],
        "Holding_Cost_Daily (TRY)": df["Holding_Cost_Daily"].round(2),
        "Ordering_Cost_Daily (TRY)": df["Ordering_Cost_Daily"].round(2),
        "Stockout_Cost_Daily (TRY)": df["Stockout_Cost_Daily"].round(2),
        "Total_Cost_Daily (TRY)": df["Total_Cost_Daily"].round(2),
    })
    
    material_name = cost_summary["material_name"].replace(" ", "_")
    material_code = cost_summary["material_code"]
    
    # Save to CSV
    csv_path = f"{output_dir}03_forecast_inventory_costs_{material_code}_{material_name}.csv"
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Inventory cost data saved to CSV → {csv_path}")
    
    # Save to Excel
    excel_path = f"{output_dir}03_forecast_inventory_costs_{material_code}_{material_name}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="Daily_Costs", index=False)
    
    print(f"✅ Inventory cost data saved to Excel → {excel_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ✅ EOQ SAW-TOOTH GRAPH (Shows negative inventory for stockout)
# ══════════════════════════════════════════════════════════════════════════════

def create_eoq_graph(df: pd.DataFrame, stats: Dict, cost_summary: Dict, output_path: str) -> None:
    """
    Clean EOQ saw-tooth graph showing stockout below zero line
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFC")

    # ─────────────────────────────────────────────────────────────────
    # 1. INVENTORY LEVEL (with negative stockout zone)
    # ─────────────────────────────────────────────────────────────────

    ax1.set_facecolor("#FFFFFF")
    days = range(len(df))

    # Safety stock zone (green)
    ax1.fill_between(days, 0, stats["rop"], alpha=0.08, color="#16A34A",
                     label="Safety Stock Zone")

    # Stockout zone (red - below 0)
    ax1.fill_between(days, 0, df["Envanter"], where=(df["Envanter"] < 0), 
                     alpha=0.15, color="#EF4444", label="Stockout Zone")

    # Plot real inventory (showing negative values)
    ax1.plot(days, df["Envanter"], color="#DC2626", linewidth=2.5,
             label="Real Inventory Level", marker="o", markersize=2,
             alpha=0.85, zorder=3)

    # ROP line
    ax1.axhline(y=stats["rop"], color="#16A34A", linestyle="--", linewidth=2.5,
                label=f"Reorder Point (ROP) = {stats['rop']:.0f}", zorder=2, alpha=0.8)

    # EOQ line
    ax1.axhline(y=stats["eoq"], color="#D97706", linestyle="--", linewidth=2,
                label=f"Order Quantity (EOQ) = {stats['eoq']:.0f}", zorder=1, alpha=0.6)

    # Average inventory
    ax1.axhline(y=stats["avg_inventory"], color="#7C3AED", linestyle=":", linewidth=2.5,
                label=f"Average Inventory = {stats['avg_inventory']:.0f}", zorder=2, alpha=0.8)

    # Zero line
    ax1.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5, zorder=0, alpha=0.5)

    ax1.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_ylabel("Inventory (units)", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_title(f"📦 EOQ SAW-TOOTH GRAPH: {stats['material_name']}\nReal Inflow & Consumption Data",
                  fontsize=13, fontweight="bold", color="#111827", pad=15)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--", color="#9CA3AF", zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    info_text = (f"📊 CONFIGURATION\n"
                f"Code: {stats['material_code']}\n"
                f"Lead Time: {stats['lead_time']} days\n"
                f"Unit Price: {stats['currency']}{stats['unit_price']:.2f}\n"
                f"Avg Daily Consumption: {stats['avg_daily_consumption']:.2f}\n"
                f"Std Deviation: {stats['std_daily']:.2f}\n"
                f"CV: {stats['cv_percent']:.1f}%\n"
                f"Service Level: {SERVICE_LEVEL*100:.0f}%\n"
                f"━━━━━━━━━━━━━━━\n"
                f"Total Cost: {stats['currency']}{cost_summary['total_cost']:,.2f}\n"
                f"Stockout Days: {cost_summary['stockout_days']}")

    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
             ha="left", va="top", fontweight="bold", color="#111827",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#FEF3C7", alpha=0.95,
                      edgecolor="#D97706", linewidth=2))

    # ─────────────────────────────────────────────────────────────────
    # 2. INFLOW vs CONSUMPTION
    # ─────────────────────────────────────────────────────────────────

    ax2.set_facecolor("#FFFFFF")

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax2.bar(x - width/2, df["Mal Giriş"], width, label="Inflow",
                    color="#16A34A", alpha=0.8, edgecolor="white")
    bars2 = ax2.bar(x + width/2, df["Tüketim Miktarı"], width, label="Consumption",
                    color="#DC2626", alpha=0.8, edgecolor="white")

    ax2.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_ylabel("Quantity (units)", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_title("Daily Inflow vs Consumption", fontsize=12, fontweight="bold",
                  color="#111827", pad=15)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    # Reduce X-axis display (many days)
    if len(df) > 50:
        step = len(df) // 10
        ax2.set_xticks(range(0, len(df), step))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#F8FAFC")
    print(f"✅ EOQ Graph saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ✅ EOQ COST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_eoq_cost_graph(stats: Dict, cost_summary: Dict, output_path: str) -> None:
    """
    EOQ Cost Function: Ordering + Holding + Total
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#F8FAFC")

    ax1.set_facecolor("#FFFFFF")

    annual_demand = stats["annual_demand"]
    eoq_value = stats["eoq"]
    unit_price = stats["unit_price"]

    Q_range = np.linspace(1, eoq_value * 3, 300)

    # Cost functions
    ordering_cost = (annual_demand / Q_range) * ORDERING_COST
    holding_cost = (Q_range / 2) * HOLDING_COST_PCT * unit_price
    total_cost = ordering_cost + holding_cost

    # Plot
    ax1.plot(Q_range, ordering_cost, linewidth=2.5, color="#DC2626",
             label="Ordering Cost", linestyle="--", alpha=0.8)
    ax1.plot(Q_range, holding_cost, linewidth=2.5, color="#16A34A",
             label="Holding Cost", linestyle="--", alpha=0.8)
    ax1.plot(Q_range, total_cost, linewidth=3.5, color="#7C3AED",
             label="Total Cost", zorder=3)

    # EOQ point
    eoq_cost = (eoq_value / 2) * HOLDING_COST_PCT * unit_price + \
               (annual_demand / eoq_value) * ORDERING_COST

    ax1.scatter([eoq_value], [eoq_cost], s=300, color="#D97706", marker="*",
                zorder=4, edgecolor="black", linewidth=2, label=f"EOQ = {eoq_value:.0f}")

    ax1.axvline(x=eoq_value, color="#D97706", linestyle=":", linewidth=2, alpha=0.7)

    ax1.set_xlabel("Order Quantity (Q)", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_ylabel(f"Annual Cost ({stats['currency']})", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_title("EOQ Cost Tradeoff Function", fontsize=12, fontweight="bold", color="#111827")
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ─────────────────────────────────────────────────────────────────
    # Cost Details
    # ─────────────────────────────────────────────────────────────────

    ax2.axis("off")

    annual_ordering = (annual_demand / eoq_value) * ORDERING_COST
    annual_holding = (eoq_value / 2) * HOLDING_COST_PCT * unit_price
    total_annual = annual_ordering + annual_holding

    cost_text = f"""
📊 EOQ COST ANALYSIS: {stats['material_name']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 DEMAND & PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Annual Demand:           {annual_demand:>12,.0f} units
Avg Daily Consumption:   {stats['avg_daily_consumption']:>12,.2f} units
Lead Time:               {stats['lead_time']:>12} days
Unit Price:              {stats['currency']}{unit_price:>12,.2f}
CV:                      {stats['cv_percent']:>12.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 EOQ (OPTIMAL ORDER QUANTITY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ordering Cost per Order: {stats['currency']}{ORDERING_COST:>12,.2f}
Holding Cost Rate:       {HOLDING_COST_PCT*100:>12.1f}%

EOQ Formula: Q* = √(2DS/h)
  D = {annual_demand:,.0f} (annual demand)
  S = {stats['currency']}{ORDERING_COST:,.2f} (ordering cost)
  h = {stats['currency']}{HOLDING_COST_PCT * unit_price:,.3f} (holding cost)

Optimal Order Quantity:  {eoq_value:>12,.0f} units

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 ANNUAL COST BREAKDOWN (THEORETICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ordering Cost:           {stats['currency']}{annual_ordering:>12,.2f}
Holding Cost:            {stats['currency']}{annual_holding:>12,.2f}
────────────────────────────────────────
TOTAL ANNUAL COST:       {stats['currency']}{total_annual:>12,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 ANNUAL COST BREAKDOWN (REAL DATA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Holding Cost:            {stats['currency']}{cost_summary['total_holding_cost'] * (365/cost_summary['days_analyzed']):>12,.2f}
Ordering Cost:           {stats['currency']}{cost_summary['total_ordering_cost'] * (365/cost_summary['days_analyzed']):>12,.2f}
Stockout Cost:           {stats['currency']}{cost_summary['total_stockout_cost'] * (365/cost_summary['days_analyzed']):>12,.2f}
────────────────────────────────────────
TOTAL ANNUAL COST:       {stats['currency']}{cost_summary['annual_cost']:>12,.2f}

Average Inventory:       {eoq_value/2:>12,.0f} units

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    ax2.text(0.05, 0.98, cost_text, transform=ax2.transAxes, fontsize=8,
             ha="left", va="top", family="monospace", color="#111827",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=1", facecolor="#FEF3C7", alpha=0.95,
                      edgecolor="#D97706", linewidth=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#F8FAFC")
    print(f"✅ EOQ Cost Graph saved → {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ✅ SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def create_summary_table(all_stats: List[Dict], all_costs: List[Dict], output_path: str) -> None:
    """
    Summary table for all materials with real costs and stockout data
    """

    df_stats = pd.DataFrame(all_stats)
    df_costs = pd.DataFrame(all_costs)
    
    # Merge stats and costs
    df_summary = df_stats.merge(df_costs[["material_code", "total_holding_cost", "total_ordering_cost", 
                                          "total_stockout_cost", "total_cost", "avg_daily_cost", "annual_cost", 
                                          "total_orders", "total_stockout_units", "stockout_days"]],
                               on="material_code")
    df_summary = df_summary.sort_values("annual_demand", ascending=False)

    # Save to CSV
    csv_path = output_path.replace(".xlsx", ".csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Summary table saved to CSV → {csv_path}")

    # Save to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Summary table saved to Excel → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_console_report(all_stats: List[Dict], all_costs: List[Dict]) -> None:
    """
    Print EOQ recommendations to console with real costs and stockout data
    """

    print(f"\n{'='*260}")
    print(f"{'🎯 EOQ OPTIMAL ORDER QUANTITY ANALYSIS WITH REAL COSTS & STOCKOUT':^260}")
    print(f"{'='*260}\n")

    print(f"{'Code':<10} {'Material':<20} {'Total Cost':>15} {'Holding Cost':>15} {'Ordering Cost':>15} "
          f"{'Stockout Cost':>15} {'Annual Cost':>15} {'Stockout Units':>15} {'Stockout Days':>12} {'EOQ':>10}")
    print(f"{'-'*260}")

    for stat in sorted(all_stats, key=lambda x: x["annual_demand"], reverse=True):
        cost = next((c for c in all_costs if c["material_code"] == stat["material_code"]), None)
        if cost:
            print(f"{stat['material_code']:<10} {stat['material_name']:<20} "
                  f"{stat['currency']}{cost['total_cost']:>14,.2f} {stat['currency']}{cost['total_holding_cost']:>14,.2f} "
                  f"{stat['currency']}{cost['total_ordering_cost']:>14,.2f} {stat['currency']}{cost['total_stockout_cost']:>14,.2f} "
                  f"{stat['currency']}{cost['annual_cost']:>14,.2f} {cost['total_stockout_units']:>15,.0f} "
                  f"{cost['stockout_days']:>12} {stat['eoq']:>10,.0f}")

    print(f"{'-'*260}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*100}")
    print(f"{'📦 EOQ ANALYSIS FROM ForecastEOQ FILE WITH REAL COSTS & STOCKOUT':^100}")
    print(f"{'='*100}\n")

    all_stats = []
    all_costs = []

    for idx, material_code in enumerate(MATERIAL_CODES, 1):
        try:
            print(f"[{idx}/{len(MATERIAL_CODES)}] 📥 Loading material {material_code}...")

            # Load data
            df, metadata = load_material_data(EXCEL_FILE, material_code)

            if df is None or metadata is None:
                print(f"⚠️  Skipping material {material_code}!\n")
                continue

            # Calculate real inventory
            df = calculate_real_inventory(df)

            # Calculate statistics
            stats = calculate_statistics(df, metadata)
            all_stats.append(stats)

            # Calculate real inventory costs
            df, cost_summary = calculate_real_inventory_costs(df, metadata)
            all_costs.append(cost_summary)

            print(f"     ✅ EOQ: {stats['eoq']:.0f} | ROP: {stats['rop']:.0f} | "
                  f"Total Cost: {stats['currency']}{cost_summary['total_cost']:,.2f}")
            if cost_summary['total_stockout_cost'] > 0:
                print(f"     ⚠️  STOCKOUT: {cost_summary['total_stockout_units']:.0f} units on {cost_summary['stockout_days']} days | Cost: {stats['currency']}{cost_summary['total_stockout_cost']:,.2f}")

            # Export inventory cost data to Excel & CSV
            print(f"     📊 Exporting inventory cost data...")
            export_inventory_cost_data(df, cost_summary, OUTPUT_DIR)

            # EOQ Graph
            print(f"     🎨 Creating EOQ graph...")
            create_eoq_graph(df, stats, cost_summary,
                           f"{OUTPUT_DIR}01_forecast_eoq_sawtooth_{material_code}_{stats['material_name'].replace(' ', '_')}.png")

            # Cost Graph
            print(f"     💰 Creating cost graph...")
            create_eoq_cost_graph(stats, cost_summary,
                                f"{OUTPUT_DIR}02_forecast_eoq_cost_{material_code}_{stats['material_name'].replace(' ', '_')}.png")

            print()

        except Exception as e:
            print(f"❌ ERROR: {material_code} → {str(e)}\n")
            continue

    # Summary table
    if all_stats and all_costs:
        print(f"\n📊 Creating summary table...")
        create_summary_table(all_stats, all_costs, f"{OUTPUT_DIR}eoq_forecast_summary.xlsx")

        # Console report
        print_console_report(all_stats, all_costs)

        print(f"{'='*100}")
        print(f"✅ ALL ANALYSIS COMPLETED!")
        print(f"{'='*100}\n")
    else:
        print(f"❌ No data loaded!")


if __name__ == "__main__":
    main()
