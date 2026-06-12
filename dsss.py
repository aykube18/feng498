import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from typing import Dict, List, Tuple
import math
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =============================================================================
# 🎨 PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Inventory DSS", layout="wide")

# =============================================================================
# 📘 SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("📘 Decision Support System")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 EDA",
        "⏳ Time Series",
        "🔮 Forecast (SARIMA + XGB + CatBoost)",
        "📦 Inventory Optimization (EOQ + ROP + Costs)",
        "📈 Dashboard"
    ]
)

# =============================================================================
# 🔧 GENERIC UTILITIES
# =============================================================================

def load_excel(uploaded_file, sheet_name="Sheet1"):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df

# =============================================================================
# 🔮 FORECASTING MODULE (SARIMA + XGBoost + CatBoost)
# =============================================================================

def make_features(series, lags=14, rolling_windows=[7, 14, 30]):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = df["y"].shift(1).rolling(w).std()
    df["dayofweek"] = series.index.dayofweek
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series))
    return df.dropna()

def recursive_forecast(model, train_series, test_index, feature_cols):
    history = list(train_series.values)
    history_idx = list(train_series.index)
    preds = []

    for i in range(len(test_index)):
        temp = pd.Series(history, index=history_idx)
        temp_df = make_features(temp)
        pred = float(model.predict(temp_df[feature_cols].iloc[[-1]])[0])
        preds.append(max(0.0, pred))
        history.append(pred)
        history_idx.append(test_index[i])

    return pd.Series(preds, index=test_index)

def run_sarima(train, test, m=7):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, m),
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    fc = fitted.forecast(steps=len(test))
    return fc.clip(lower=0)

def run_xgboost(train, test):
    df_train = make_features(train)
    feature_cols = [c for c in df_train.columns if c != "y"]
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(df_train[feature_cols], df_train["y"])
    return recursive_forecast(model, train, test.index, feature_cols)

def run_catboost(train, test):
    df_train = make_features(train)
    feature_cols = [c for c in df_train.columns if c != "y"]
    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    )
    model.fit(df_train[feature_cols], df_train["y"])
    return recursive_forecast(model, train, test.index, feature_cols)

def model_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(
        np.abs((actual.values - forecast.values) / np.where(actual.values == 0, 1, actual.values))
    ) * 100
    return mae, rmse, mape

# =============================================================================
# 📦 INVENTORY MODULE (NEW VERSION WITH STOCKOUT & COSTS)
# =============================================================================

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

ORDERING_COST = 2792
HOLDING_COST_PCT = 0.25
SERVICE_LEVEL = 0.95
STOCKOUT_COST_PER_UNIT = 500

def load_material_data(uploaded_file, material_code: int) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_excel(uploaded_file, sheet_name=str(material_code))
    df.columns = df.columns.str.strip()
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih").reset_index(drop=True)
    df["Mal Giriş"] = df["Mal Giriş"].fillna(0)
    df["Tüketim Miktarı"] = df["Tüketim Miktarı"].fillna(0)

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

def calculate_real_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Envanter"] = 0.0
    df["Stockout"] = 0.0

    current_inv = 0.0
    inventory_list = []
    stockout_list = []

    for _, row in df.iterrows():
        current_inv = current_inv + row["Mal Giriş"] - row["Tüketim Miktarı"]
        if current_inv < 0:
            stockout_list.append(-current_inv)
        else:
            stockout_list.append(0)
        inventory_list.append(current_inv)

    df["Envanter"] = inventory_list
    df["Stockout"] = stockout_list
    return df

def calculate_real_inventory_costs(df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    unit_price = metadata["unit_price"]
    daily_holding_cost_rate = HOLDING_COST_PCT / 365

    df["Holding_Cost_Daily"] = df["Envanter"].clip(lower=0) * daily_holding_cost_rate * unit_price
    df["Is_Order"] = df["Mal Giriş"] > 0
    df["Ordering_Cost_Daily"] = df["Is_Order"] * ORDERING_COST
    df["Stockout_Cost_Daily"] = df["Stockout"] * STOCKOUT_COST_PER_UNIT
    df["Total_Cost_Daily"] = (
        df["Holding_Cost_Daily"] + df["Ordering_Cost_Daily"] + df["Stockout_Cost_Daily"]
    )

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
        "avg_order_qty": round(df[df["Mal Giriş"] > 0]["Mal Giriş"].mean(), 2)
        if total_orders > 0
        else 0,
        "total_stockout_units": round(total_stockout_units, 2),
        "stockout_days": int(stockout_days),
        "unit_price": metadata["unit_price"],
        "currency": metadata["currency"],
    }
    return df, cost_summary

def calculate_statistics(df: pd.DataFrame, metadata: Dict) -> Dict:
    consumption = df["Tüketim Miktarı"].values
    inflow = df["Mal Giriş"].values
    inventory = df["Envanter"].values

    days = len(df)
    total_consumption = consumption.sum()
    total_inflow = inflow.sum()
    avg_daily_consumption = total_consumption / days if days > 0 else 0

    consumption_nonzero = consumption[consumption > 0]
    std_consumption = (
        consumption_nonzero.std() if len(consumption_nonzero) > 1 else avg_daily_consumption * 0.3
    )
    std_daily = std_consumption / np.sqrt(len(df)) if len(df) > 1 else avg_daily_consumption * 0.3
    cv = (std_daily / avg_daily_consumption * 100) if avg_daily_consumption > 0 else 0

    annual_demand = avg_daily_consumption * 365
    holding_cost = HOLDING_COST_PCT * metadata["unit_price"]

    if annual_demand > 0 and holding_cost > 0:
        eoq = math.sqrt((2 * annual_demand * ORDERING_COST) / holding_cost)
    else:
        eoq = avg_daily_consumption * 30
    eoq = max(1.0, eoq)

    lead_time = metadata["lead_time"]
    lt_mean = avg_daily_consumption * lead_time
    lt_std = std_daily * math.sqrt(lead_time)
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

def create_eoq_graph_fig(df: pd.DataFrame, stats: Dict, cost_summary: Dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFC")

    ax1.set_facecolor("#FFFFFF")
    days = range(len(df))

    ax1.fill_between(days, 0, stats["rop"], alpha=0.08, color="#16A34A", label="Safety Stock Zone")
    ax1.fill_between(
        days,
        0,
        df["Envanter"],
        where=(df["Envanter"] < 0),
        alpha=0.15,
        color="#EF4444",
        label="Stockout Zone",
    )

    ax1.plot(
        days,
        df["Envanter"],
        color="#DC2626",
        linewidth=2.5,
        label="Real Inventory Level",
        marker="o",
        markersize=2,
        alpha=0.85,
        zorder=3,
    )

    ax1.axhline(
        y=stats["rop"],
        color="#16A34A",
        linestyle="--",
        linewidth=2.5,
        label=f"Reorder Point (ROP) = {stats['rop']:.0f}",
        zorder=2,
        alpha=0.8,
    )
    ax1.axhline(
        y=stats["eoq"],
        color="#D97706",
        linestyle="--",
        linewidth=2,
        label=f"Order Quantity (EOQ) = {stats['eoq']:.0f}",
        zorder=1,
        alpha=0.6,
    )
    ax1.axhline(
        y=stats["avg_inventory"],
        color="#7C3AED",
        linestyle=":",
        linewidth=2.5,
        label=f"Average Inventory = {stats['avg_inventory']:.0f}",
        zorder=2,
        alpha=0.8,
    )
    ax1.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5, zorder=0, alpha=0.5)

    ax1.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_ylabel("Inventory (units)", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_title(
        f"📦 EOQ SAW-TOOTH GRAPH: {stats['material_name']}\nReal Inflow & Consumption Data",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        pad=15,
    )
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--", color="#9CA3AF", zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    info_text = (
        f"📊 CONFIGURATION\n"
        f"Code: {stats['material_code']}\n"
        f"Lead Time: {stats['lead_time']} days\n"
        f"Unit Price: {stats['currency']}{stats['unit_price']:.2f}\n"
        f"Avg Daily Consumption: {stats['avg_daily_consumption']:.2f}\n"
        f"Std Deviation: {stats['std_daily']:.2f}\n"
        f"CV: {stats['cv_percent']:.1f}%\n"
        f"Service Level: {SERVICE_LEVEL*100:.0f}%\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Total Cost: {stats['currency']}{cost_summary['total_cost']:,.2f}\n"
        f"Stockout Days: {cost_summary['stockout_days']}"
    )

    ax1.text(
        0.02,
        0.98,
        info_text,
        transform=ax1.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        fontweight="bold",
        color="#111827",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#FEF3C7",
            alpha=0.95,
            edgecolor="#D97706",
            linewidth=2,
        ),
    )

    ax2.set_facecolor("#FFFFFF")
    x = np.arange(len(df))
    width = 0.35

    ax2.bar(
        x - width / 2,
        df["Mal Giriş"],
        width,
        label="Inflow",
        color="#16A34A",
        alpha=0.8,
        edgecolor="white",
    )
    ax2.bar(
        x + width / 2,
        df["Tüketim Miktarı"],
        width,
        label="Consumption",
        color="#DC2626",
        alpha=0.8,
        edgecolor="white",
    )

    ax2.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_ylabel("Quantity (units)", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_title(
        "Daily Inflow vs Consumption", fontsize=12, fontweight="bold", color="#111827", pad=15
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    if len(df) > 50:
        step = len(df) // 10
        ax2.set_xticks(range(0, len(df), step))

    plt.tight_layout()
    return fig

def create_eoq_cost_graph_fig(stats: Dict, cost_summary: Dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#F8FAFC")
    ax1.set_facecolor("#FFFFFF")

    annual_demand = stats["annual_demand"]
    eoq_value = stats["eoq"]
    unit_price = stats["unit_price"]

    Q_range = np.linspace(1, eoq_value * 3, 300)
    ordering_cost = (annual_demand / Q_range) * ORDERING_COST
    holding_cost = (Q_range / 2) * HOLDING_COST_PCT * unit_price
    total_cost = ordering_cost + holding_cost

    ax1.plot(
        Q_range,
        ordering_cost,
        linewidth=2.5,
        color="#DC2626",
        label="Ordering Cost",
        linestyle="--",
        alpha=0.8,
    )
    ax1.plot(
        Q_range,
        holding_cost,
        linewidth=2.5,
        color="#16A34A",
        label="Holding Cost",
        linestyle="--",
        alpha=0.8,
    )
    ax1.plot(
        Q_range,
        total_cost,
        linewidth=3.5,
        color="#7C3AED",
        label="Total Cost",
        zorder=3,
    )

    eoq_cost = (eoq_value / 2) * HOLDING_COST_PCT * unit_price + (
        annual_demand / eoq_value
    ) * ORDERING_COST

    ax1.scatter(
        [eoq_value],
        [eoq_cost],
        s=300,
        color="#D97706",
        marker="*",
        zorder=4,
        edgecolor="black",
        linewidth=2,
        label=f"EOQ = {eoq_value:.0f}",
    )
    ax1.axvline(x=eoq_value, color="#D97706", linestyle=":", linewidth=2, alpha=0.7)

    ax1.set_xlabel("Order Quantity (Q)", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_ylabel(f"Annual Cost ({stats['currency']})", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_title("EOQ Cost Tradeoff Function", fontsize=12, fontweight="bold", color="#111827")
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

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
    ax2.text(
        0.05,
        0.98,
        cost_text,
        transform=ax2.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        family="monospace",
        color="#111827",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="#FEF3C7",
            alpha=0.95,
            edgecolor="#D97706",
            linewidth=2,
        ),
    )

    plt.tight_layout()
    return fig

# =============================================================================
# 📊 PAGE 1 — EDA
# =============================================================================

if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    uploaded = st.file_uploader("Upload any Excel file", type=["xlsx"])
    if uploaded:
        sheet_names = pd.ExcelFile(uploaded).sheet_names
        sheet = st.selectbox("Select sheet", sheet_names)
        df = load_excel(uploaded, sheet)
        st.subheader("📄 Preview")
        st.dataframe(df)
        st.subheader("📈 Summary statistics")
        st.write(df.describe())

# =============================================================================
# ⏳ PAGE 2 — TIME SERIES
# =============================================================================

elif page == "⏳ Time Series":
    st.title("⏳ Time Series Visualization")
    uploaded = st.file_uploader("Upload time series Excel (Date, Consumption)", type=["xlsx"])
    if uploaded:
        df = load_excel(uploaded, "Sheet1")
        df.columns = ["Date", "Consumption"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["Date"], df["Consumption"], color="#2563EB")
        ax.set_title("Daily Consumption")
        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        st.pyplot(fig)

# =============================================================================
# 🔮 PAGE 3 — FORECASTING
# =============================================================================

elif page == "🔮 Forecast (SARIMA + XGB + CatBoost)":
    st.title("🔮 Forecasting — SARIMA vs XGBoost vs CatBoost")
    uploaded = st.file_uploader("Upload motorin_tuketim.xlsx (Date, Consumption)", type=["xlsx"])

    if uploaded:
        df = load_excel(uploaded, "Sheet1")
        df.columns = ["Date", "Consumption"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        series = df.set_index("Date")["Consumption"]
        full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
        series = series.reindex(full_idx, fill_value=0)

        split = int(len(series) * 0.8)
        train, test = series.iloc[:split], series.iloc[split:]

        st.subheader("📌 Running models...")
        fc_sarima = run_sarima(train, test)
        fc_xgb = run_xgboost(train, test)
        fc_cat = run_catboost(train, test)

        m_sarima = model_metrics(test, fc_sarima)
        m_xgb = model_metrics(test, fc_xgb)
        m_cat = model_metrics(test, fc_cat)

        c1, c2, c3 = st.columns(3)
        c1.metric("SARIMA MAPE", f"{m_sarima[2]:.1f}%")
        c2.metric("XGBoost MAPE", f"{m_xgb[2]:.1f}%")
        c3.metric("CatBoost MAPE", f"{m_cat[2]:.1f}%")

        st.subheader("📈 Model comparison")
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(series.index, series.values, label="Actual", color="black", linewidth=1)
        ax.plot(fc_sarima.index, fc_sarima.values, label="SARIMA", linestyle="--", color="#e53935")
        ax.plot(fc_xgb.index, fc_xgb.values, label="XGBoost", linestyle="-.", color="#43a047")
        ax.plot(fc_cat.index, fc_cat.values, label="CatBoost", linestyle=":", color="#8e24aa")
        ax.axvspan(test.index[0], test.index[-1], alpha=0.1, color="gray", label="Test region")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Consumption")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

# =============================================================================
# 📦 PAGE 4 — INVENTORY OPTIMIZATION
# =============================================================================

elif page == "📦 Inventory Optimization (EOQ + ROP + Costs)":
    st.title("📦 Inventory Optimization — EOQ, ROP, Stockout & Cost Analysis")

    uploaded = st.file_uploader("Upload ForecastEOQ.xlsx", type=["xlsx"])
    if uploaded:
        material_code = st.selectbox("Select material code", MATERIAL_CODES)
        df_raw, metadata = load_material_data(uploaded, material_code)
        df_inv = calculate_real_inventory(df_raw)
        stats_dict = calculate_statistics(df_inv, metadata)
        df_cost, cost_summary = calculate_real_inventory_costs(df_inv, metadata)

        st.subheader("📌 Key indicators")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EOQ", f"{stats_dict['eoq']:.0f}")
        c2.metric("ROP", f"{stats_dict['rop']:.0f}")
        c3.metric("Annual Demand", f"{stats_dict['annual_demand']:.0f}")
        c4.metric(
            "Annual Cost",
            f"{cost_summary['annual_cost']:,.0f} {stats_dict['currency']}",
        )

        st.subheader("📦 EOQ saw-tooth graph")
        fig1 = create_eoq_graph_fig(df_inv, stats_dict, cost_summary)
        st.pyplot(fig1)

        st.subheader("💰 EOQ cost function")
        fig2 = create_eoq_cost_graph_fig(stats_dict, cost_summary)
        st.pyplot(fig2)

        st.subheader("📄 Daily inventory & cost table")
        st.dataframe(df_cost)

# =============================================================================
# 📈 PAGE 5 — DASHBOARD
# =============================================================================

elif page == "📈 Dashboard":
    st.title("📈 Executive Dashboard")
    st.info("High-level KPIs and combined visualizations can be added here later.")
