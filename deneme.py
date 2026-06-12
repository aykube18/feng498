import io
import warnings
import itertools
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
st.set_page_config(
    page_title="Integrated Decision Support System",
    layout="wide",
    page_icon="📦",
)

DATE_COL = "Tarih"
VALUE_COL = "Tüketim"

TEST_START = "2025-04-02"  # first predicted day (prediction starts here)
SEASONAL = 7               # weekly seasonality for daily data
DEFAULT_ORDER = (1, 1, 2)
DEFAULT_SORDER = (1, 1, 1, SEASONAL)
ALPHA = 0.05
AUTO_SELECT = False

N_LAGS = 14
ROLL_WINDOWS = (7, 14, 30)

ORDERING_COST = 2792          # TRY
HOLDING_COST_PCT = 0.25       # 25% annual
SERVICE_LEVEL = 0.95
STOCKOUT_COST_PER_UNIT = 500  # TRY

MATERIAL_CODES = [600080, 600096, 600102, 600112, 603789,
                  603812, 607290, 626100, 627518, 627519]

SHORT_NAMES = {
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

# =============================================================================
# HEADER + DATA UPLOAD
# =============================================================================
st.markdown(
    "<h1 style='color:#111827;'>📦 Integrated Decision Support System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='color:#374151;'>1️⃣ Data Upload</h3>",
    unsafe_allow_html=True,
)

col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown("#### 📂 Inventory Data (ForecastEOQ.xlsx)")
    inv_file = st.file_uploader(
        "Upload inventory Excel file",
        type=["xlsx", "xls"],
        key="inventory_upload",
    )

with col_up2:
    st.markdown("#### 📈 Forecast Data (motorin_tuketim.xlsx)")
    fc_file = st.file_uploader(
        "Upload forecast Excel file",
        type=["xlsx", "xls"],
        key="forecast_upload",
    )

if inv_file is not None:
    st.success("✅ Inventory data file loaded.")
else:
    st.info("Upload inventory file for EOQ & cost analysis.")

if fc_file is not None:
    st.success("✅ Forecast data file loaded.")
else:
    st.info("Upload forecast file for SARIMA + XGBoost + CatBoost comparison.")

# =============================================================================
# TABS
# =============================================================================
tab_eda, tab_abcxyz, tab_forecast, tab_inventory, tab_ts, tab_summary = st.tabs(
    [
        "📊 EDA",
        "🔤 ABC–XYZ",
        "📈 Forecast",
        "📦 Inventory",
        "📉 Time Series",
        "📋 Summary Dashboard",
    ]
)

# =============================================================================
# INVENTORY FUNCTIONS
# =============================================================================
def load_material_data(file_obj, material_code: int):
    try:
        df = pd.read_excel(file_obj, sheet_name=str(material_code))
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
    except Exception as e:
        st.error(f"Error reading material {material_code}: {e}")
        return None, None


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


def calculate_real_inventory_costs(df: pd.DataFrame, metadata: dict):
    df = df.copy()
    unit_price = metadata["unit_price"]
    daily_holding_cost_rate = HOLDING_COST_PCT / 365

    df["Holding_Cost_Daily"] = df["Envanter"].clip(lower=0) * daily_holding_cost_rate * unit_price
    df["Is_Order"] = df["Mal Giriş"] > 0
    df["Ordering_Cost_Daily"] = df["Is_Order"] * ORDERING_COST
    df["Stockout_Cost_Daily"] = df["Stockout"] * STOCKOUT_COST_PER_UNIT
    df["Total_Cost_Daily"] = (
        df["Holding_Cost_Daily"]
        + df["Ordering_Cost_Daily"]
        + df["Stockout_Cost_Daily"]
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
        "avg_order_qty": round(
            df[df["Mal Giriş"] > 0]["Mal Giriş"].mean(), 2
        )
        if total_orders > 0
        else 0,
        "total_stockout_units": round(total_stockout_units, 2),
        "stockout_days": int(stockout_days),
        "unit_price": metadata["unit_price"],
        "currency": metadata["currency"],
    }
    return df, cost_summary


def calculate_statistics(df: pd.DataFrame, metadata: dict):
    consumption = df["Tüketim Miktarı"].values
    inflow = df["Mal Giriş"].values
    inventory = df["Envanter"].values

    days = len(df)
    total_consumption = consumption.sum()
    total_inflow = inflow.sum()
    avg_daily_consumption = total_consumption / days if days > 0 else 0

    consumption_nonzero = consumption[consumption > 0]
    std_consumption = (
        consumption_nonzero.std()
        if len(consumption_nonzero) > 1
        else avg_daily_consumption * 0.3
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


def create_eoq_graph(df, stats, cost_summary):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFC")

    ax1.set_facecolor("#FFFFFF")
    days = range(len(df))

    ax1.fill_between(
        days,
        0,
        stats["rop"],
        alpha=0.08,
        color="#16A34A",
        label="Safety Stock Zone",
    )

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
    ax1.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5, alpha=0.5)

    ax1.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_ylabel("Inventory (units)", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_title(
        f"📦 EOQ Saw-Tooth Graph: {stats['material_name']}",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        pad=15,
    )
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--", color="#9CA3AF")
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
        "Daily Inflow vs Consumption",
        fontsize=12,
        fontweight="bold",
        color="#111827",
        pad=15,
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    if len(df) > 50:
        step = len(df) // 10
        ax2.set_xticks(range(0, len(df), step))

    plt.tight_layout()
    return fig


def create_eoq_cost_graph(stats, cost_summary):
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
    ax1.set_ylabel(
        f"Annual Cost ({stats['currency']})",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )
    ax1.set_title(
        "EOQ Cost Tradeoff Function",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )
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
# FORECAST FUNCTIONS (forecast_eren_11_06 style)
# =============================================================================
def load_daily(path_or_buffer) -> pd.DataFrame:
    df = pd.read_excel(path_or_buffer)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    daily = (
        df.set_index(DATE_COL)[VALUE_COL]
        .sort_index()
        .asfreq("D")
        .fillna(0)
    )
    out = pd.DataFrame({"date": daily.index, VALUE_COL: daily.values})
    out["t"] = np.arange(1, len(out) + 1)
    return out


def split_train_test(frame: pd.DataFrame):
    cutoff = pd.Timestamp(TEST_START)
    train = frame[frame["date"] < cutoff].reset_index(drop=True)
    test = frame[frame["date"] >= cutoff].reset_index(drop=True)
    return train, test


def _aicc(result) -> float:
    k = len(result.params)
    n = result.nobs
    return result.aic + (2 * k * k + 2 * k) / (n - k - 1) if (n - k - 1) > 0 else np.inf


def select_order_aicc(y: pd.Series):
    best = {"score": np.inf, "order": DEFAULT_ORDER, "sorder": DEFAULT_SORDER}
    for p, d, q in itertools.product(range(3), range(2), range(3)):
        for P, D, Q in itertools.product(range(2), range(2), range(2)):
            try:
                res = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, SEASONAL),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                score = _aicc(res)
                if np.isfinite(score) and score < best["score"]:
                    best = {
                        "score": score,
                        "order": (p, d, q),
                        "sorder": (P, D, Q, SEASONAL),
                    }
            except Exception:
                continue
    return best["order"], best["sorder"]


def rolling_one_step(y_train: pd.Series, test_values: np.ndarray, order, sorder):
    res = SARIMAX(
        y_train,
        order=order,
        seasonal_order=sorder,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

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

    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)
    return pred, lo, hi


def make_feature_row(history_values, idx_date, t_value):
    h = np.asarray(history_values, dtype=float)
    feat = {}
    for lag in range(1, N_LAGS + 1):
        feat[f"lag_{lag}"] = h[-lag] if len(h) >= lag else 0.0
    for w in ROLL_WINDOWS:
        window = h[-w:] if len(h) >= 1 else np.array([0.0])
        feat[f"rmean_{w}"] = float(np.mean(window)) if len(window) else 0.0
        feat[f"rstd_{w}"] = float(np.std(window)) if len(window) > 1 else 0.0

    dow = idx_date.dayofweek
    feat["dow"] = dow
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feat["month"] = idx_date.month
    feat["month_sin"] = np.sin(2 * np.pi * idx_date.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * idx_date.month / 12)
    feat["trend"] = t_value
    return feat


def build_training_matrix(train_frame):
    values = train_frame[VALUE_COL].values.astype(float)
    dates = train_frame["date"].values
    ts = train_frame["t"].values
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
        feat = make_feature_row(history, pd.Timestamp(row["date"]), int(row["t"]))
        x = pd.DataFrame([feat])[feature_cols]
        yhat = max(0.0, float(model.predict(x)[0]))
        preds.append(yhat)
        history.append(float(row[VALUE_COL]))
    return np.asarray(preds)


def fit_xgboost(X, y):
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
    model.fit(X, y, verbose=False)
    return model


def fit_catboost(X, y):
    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    )
    model.fit(X, y)
    return model


def error_stats(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100) if mask.any() else np.nan
    return mae, rmse, mape


def build_comparison(test, sarima_pred, lo, hi, xgb_pred, cat_pred):
    out = pd.DataFrame(
        {
            "date": test["date"].values,
            "t": test["t"].values,
            "Actual": test[VALUE_COL].values.astype(float),
            "SARIMA": sarima_pred,
            "SARIMA_Lo95": lo,
            "SARIMA_Hi95": hi,
            "XGBoost": xgb_pred,
            "CatBoost": cat_pred,
        }
    )
    return out


def plot_results(frame, train, comp, order, sorder, stats):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    boundary = train["date"].iloc[-1]

    ax1.plot(
        frame["date"],
        frame[VALUE_COL],
        color="#1f77b4",
        linewidth=0.7,
        alpha=0.55,
        label="Actual (daily)",
    )
    ax1.plot(
        comp["date"],
        comp["SARIMA"],
        color="#2ca02c",
        linewidth=1.3,
        label="SARIMA",
    )
    ax1.plot(
        comp["date"],
        comp["XGBoost"],
        color="#d62728",
        linewidth=1.3,
        linestyle="--",
        label="XGBoost",
    )
    ax1.plot(
        comp["date"],
        comp["CatBoost"],
        color="#9467bd",
        linewidth=1.3,
        linestyle="-.",
        label="CatBoost",
    )

    ax1.axvline(boundary, color="grey", linestyle=":", linewidth=1.3)
    ax1.text(
        boundary,
        ax1.get_ylim()[1],
        " train | test",
        color="grey",
        va="top",
        ha="left",
        fontsize=9,
    )

    ax1.set_title(
        "Full daily series — three models",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Daily consumption")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax2.fill_between(
        comp["date"],
        comp["SARIMA_Lo95"],
        comp["SARIMA_Hi95"],
        color="#2ca02c",
        alpha=0.10,
        label="SARIMA 95% interval",
    )
    ax2.plot(
        comp["date"],
        comp["Actual"],
        color="#1f77b4",
        linewidth=1.0,
        marker="o",
        markersize=2.5,
        label="Actual (daily)",
    )
    ax2.plot(
        comp["date"],
        comp["SARIMA"],
        color="#2ca02c",
        linewidth=1.6,
        label="SARIMA",
    )
    ax2.plot(
        comp["date"],
        comp["XGBoost"],
        color="#d62728",
        linewidth=1.6,
        linestyle="--",
        label="XGBoost",
    )
    ax2.plot(
        comp["date"],
        comp["CatBoost"],
        color="#9467bd",
        linewidth=1.6,
        linestyle="-.",
        label="CatBoost",
    )

    pstart = pd.Timestamp(comp["date"].iloc[0]).strftime("%Y-%m-%d")
    ax2.set_title(
        f"Prediction window from {pstart}: real vs rolling predictions (daily)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Date (modeled as t = 1, 2, 3, … , N)")
    ax2.set_ylabel("Daily consumption")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    sub = (
        f"SARIMA MAE={stats['SARIMA'][0]:,.0f} RMSE={stats['SARIMA'][1]:,.0f} | "
        f"XGBoost MAE={stats['XGBoost'][0]:,.0f} RMSE={stats['XGBoost'][1]:,.0f} | "
        f"CatBoost MAE={stats['CatBoost'][0]:,.0f} RMSE={stats['CatBoost'][1]:,.0f}"
    )

    fig.suptitle(
        f"Daily Motorin Consumption — SARIMA{order}{sorder[:3]}s={sorder[3]} "
        f"vs XGBoost vs CatBoost (rolling 1-step)\n{sub}",
        fontsize=12,
        fontweight="bold",
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# TAB: EDA
# =============================================================================
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")
    if inv_file is None:
        st.info("Upload inventory Excel file to see EDA.")
    else:
        xls = pd.ExcelFile(inv_file)
        sheet = st.selectbox("Select sheet for EDA", xls.sheet_names)
        df_eda = pd.read_excel(inv_file, sheet_name=sheet)
        st.write("#### Preview")
        st.dataframe(df_eda.head())
        st.write("#### Summary Statistics")
        st.dataframe(df_eda.describe(include="all"))

# =============================================================================
# TAB: ABC–XYZ (placeholder)
# =============================================================================
with tab_abcxyz:
    st.markdown("### 🔤 ABC–XYZ Classification")
    st.info("ABC–XYZ logic can be plugged here using annual demand and CV from inventory stats.")

# =============================================================================
# TAB: FORECAST
# =============================================================================
with tab_forecast:
    st.markdown("### 📈 Forecast — SARIMA + XGBoost + CatBoost (Daily Motorin)")
    if fc_file is None:
        st.info("Upload forecast file (motorin_tuketim.xlsx) to run models.")
    else:
        frame = load_daily(fc_file)
        train, test = split_train_test(frame)

        y_train = pd.Series(
            train[VALUE_COL].values,
            index=pd.RangeIndex(1, len(train) + 1),
            name=VALUE_COL,
        )

        if AUTO_SELECT:
            order, sorder = select_order_aicc(y_train)
        else:
            order, sorder = DEFAULT_ORDER, DEFAULT_SORDER

        sarima_pred, lo, hi = rolling_one_step(
            y_train, test[VALUE_COL].values.astype(float), order, sorder
        )

        X_tr, y_tr, feature_cols = build_training_matrix(train)
        xgb_model = fit_xgboost(X_tr, y_tr)
        cat_model = fit_catboost(X_tr, y_tr)

        xgb_pred = rolling_one_step_ml(xgb_model, train, test, feature_cols)
        cat_pred = rolling_one_step_ml(cat_model, train, test, feature_cols)

        comp = build_comparison(test, sarima_pred, lo, hi, xgb_pred, cat_pred)
        stats_fc = {
            "SARIMA": error_stats(comp["Actual"], comp["SARIMA"]),
            "XGBoost": error_stats(comp["Actual"], comp["XGBoost"]),
            "CatBoost": error_stats(comp["Actual"], comp["CatBoost"]),
        }

        col_top, col_table = st.columns([2, 1])
        with col_top:
            fig_fc = plot_results(frame, train, comp, order, sorder, stats_fc)
            st.pyplot(fig_fc)

        with col_table:
            st.markdown("#### 📋 Test Window Comparison Table")
            st.dataframe(comp)

        st.markdown("#### 📊 Model Accuracy (Test Set)")
        col_m1, col_m2, col_m3 = st.columns(3)
        for col, name in zip(
            [col_m1, col_m2, col_m3], ["SARIMA", "XGBoost", "CatBoost"]
        ):
            mae, rmse, mape = stats_fc[name]
            with col:
                st.metric(
                    label=f"{name} MAE",
                    value=f"{mae:,.0f}",
                )
                st.metric(
                    label=f"{name} RMSE",
                    value=f"{rmse:,.0f}",
                )
                st.metric(
                    label=f"{name} MAPE (%)",
                    value=f"{mape:,.1f}",
                )

# =============================================================================
# TAB: INVENTORY
# =============================================================================
with tab_inventory:
    st.markdown("### 📦 Inventory — EOQ, ROP, Cost & Stockout")
    if inv_file is None:
        st.info("Upload inventory Excel file to analyze EOQ & costs.")
    else:
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            material_code = st.selectbox(
                "Select Material Code",
                MATERIAL_CODES,
                format_func=lambda x: f"{x} - {SHORT_NAMES.get(x, str(x))}",
            )

        df_mat, metadata = load_material_data(inv_file, material_code)
        if df_mat is not None:
            df_inv = calculate_real_inventory(df_mat)
            df_inv_cost, cost_summary = calculate_real_inventory_costs(
                df_inv, metadata
            )
            stats_inv = calculate_statistics(df_inv, metadata)

            with col_info:
                st.markdown("#### 🔍 Key Inventory Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Avg Daily Consumption",
                    f"{stats_inv['avg_daily_consumption']:.2f}",
                )
                c2.metric("EOQ", f"{stats_inv['eoq']:.0f}")
                c3.metric("ROP", f"{stats_inv['rop']:.0f}")

                c4, c5, c6 = st.columns(3)
                c4.metric(
                    "Total Cost (Period)",
                    f"{cost_summary['currency']}{cost_summary['total_cost']:,.0f}",
                )
                c5.metric(
                    "Annual Cost (Real)",
                    f"{cost_summary['currency']}{cost_summary['annual_cost']:,.0f}",
                )
                c6.metric(
                    "Stockout Units",
                    f"{cost_summary['total_stockout_units']:,.0f}",
                )

            st.markdown("#### 📉 EOQ Saw-Tooth & Inflow vs Consumption")
            fig_eoq = create_eoq_graph(df_inv_cost, stats_inv, cost_summary)
            st.pyplot(fig_eoq)

            st.markdown("#### 💰 EOQ Cost Tradeoff")
            fig_cost = create_eoq_cost_graph(stats_inv, cost_summary)
            st.pyplot(fig_cost)

            st.markdown("#### 📋 Daily Inventory & Cost Table")
            st.dataframe(
                df_inv_cost[
                    [
                        "Tarih",
                        "Mal Giriş",
                        "Tüketim Miktarı",
                        "Envanter",
                        "Stockout",
                        "Holding_Cost_Daily",
                        "Ordering_Cost_Daily",
                        "Stockout_Cost_Daily",
                        "Total_Cost_Daily",
                    ]
                ]
            )

# =============================================================================
# TAB: TIME SERIES
# =============================================================================
with tab_ts:
    st.markdown("### 📉 Time Series — Inventory Consumption")
    if inv_file is None:
        st.info("Upload inventory Excel file to see time series.")
    else:
        mat_code_ts = st.selectbox(
            "Select Material for Time Series",
            MATERIAL_CODES,
            format_func=lambda x: f"{x} - {SHORT_NAMES.get(x, str(x))}",
            key="ts_material",
        )
        df_ts, meta_ts = load_material_data(inv_file, mat_code_ts)
        if df_ts is not None:
            fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
            ax_ts.plot(
                df_ts["Tarih"],
                df_ts["Tüketim Miktarı"],
                color="#1f77b4",
                linewidth=1.2,
            )
            ax_ts.set_title(
                f"Daily Consumption — {meta_ts['material_name']}",
                fontsize=11,
                fontweight="bold",
            )
            ax_ts.set_xlabel("Date")
            ax_ts.set_ylabel("Consumption")
            ax_ts.grid(True, alpha=0.3)
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts)

            st.markdown("#### 📋 Raw Data")
            st.dataframe(df_ts)

# =============================================================================
# TAB: SUMMARY DASHBOARD
# =============================================================================
with tab_summary:
    st.markdown("### 📋 Summary Dashboard — Inventory & Forecast")
    if inv_file is None:
        st.info("Upload inventory Excel file to build summary dashboard.")
    else:
        all_stats = []
        all_costs = []
        for code in MATERIAL_CODES:
            df_m, meta_m = load_material_data(inv_file, code)
            if df_m is None:
                continue
            df_i = calculate_real_inventory(df_m)
            df_i_cost, cost_m = calculate_real_inventory_costs(df_i, meta_m)
            stats_m = calculate_statistics(df_i, meta_m)
            all_stats.append(stats_m)
            all_costs.append(cost_m)

        if all_stats and all_costs:
            df_stats = pd.DataFrame(all_stats)
            df_costs = pd.DataFrame(all_costs)
            df_summary = df_stats.merge(
                df_costs[
                    [
                        "material_code",
                        "total_holding_cost",
                        "total_ordering_cost",
                        "total_stockout_cost",
                        "total_cost",
                        "avg_daily_cost",
                        "annual_cost",
                        "total_orders",
                        "total_stockout_units",
                        "stockout_days",
                    ]
                ],
                on="material_code",
            ).sort_values("annual_demand", ascending=False)

            st.markdown("#### 📊 Inventory Summary Table")
            st.dataframe(df_summary)

            st.markdown("#### 🔝 Top Materials by Annual Demand")
            top_n = st.slider("Top N", 3, 10, 5)
            df_top = df_summary.head(top_n)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                ax_bar.bar(
                    df_top["material_name"],
                    df_top["annual_demand"],
                    color="#2563EB",
                )
                ax_bar.set_title("Annual Demand (Top N)")
                ax_bar.set_ylabel("Units")
                ax_bar.tick_params(axis="x", rotation=45)
                st.pyplot(fig_bar)

            with col_chart2:
                fig_cost_bar, ax_cost_bar = plt.subplots(figsize=(6, 4))
                ax_cost_bar.bar(
                    df_top["material_name"],
                    df_top["annual_cost"],
                    color="#DC2626",
                )
                ax_cost_bar.set_title("Annual Cost (Real, Top N)")
                ax_cost_bar.set_ylabel("TRY")
                ax_cost_bar.tick_params(axis="x", rotation=45)
                st.pyplot(fig_cost_bar)

    if fc_file is not None:
        st.markdown("#### 📈 Forecast Summary (Motorin)")
        frame_fc = load_daily(fc_file)
        train_fc, test_fc = split_train_test(frame_fc)
        y_train_fc = pd.Series(
            train_fc[VALUE_COL].values,
            index=pd.RangeIndex(1, len(train_fc) + 1),
            name=VALUE_COL,
        )
        if AUTO_SELECT:
            order_fc, sorder_fc = select_order_aicc(y_train_fc)
        else:
            order_fc, sorder_fc = DEFAULT_ORDER, DEFAULT_SORDER

        sarima_pred_fc, lo_fc, hi_fc = rolling_one_step(
            y_train_fc, test_fc[VALUE_COL].values.astype(float), order_fc, sorder_fc
        )
        X_tr_fc, y_tr_fc, feature_cols_fc = build_training_matrix(train_fc)
        xgb_model_fc = fit_xgboost(X_tr_fc, y_tr_fc)
        cat_model_fc = fit_catboost(X_tr_fc, y_tr_fc)
        xgb_pred_fc = rolling_one_step_ml(
            xgb_model_fc, train_fc, test_fc, feature_cols_fc
        )
        cat_pred_fc = rolling_one_step_ml(
            cat_model_fc, train_fc, test_fc, feature_cols_fc
        )
        comp_fc = build_comparison(
            test_fc, sarima_pred_fc, lo_fc, hi_fc, xgb_pred_fc, cat_pred_fc
        )
        stats_fc2 = {
            "SARIMA": error_stats(comp_fc["Actual"], comp_fc["SARIMA"]),
            "XGBoost": error_stats(comp_fc["Actual"], comp_fc["XGBoost"]),
            "CatBoost": error_stats(comp_fc["Actual"], comp_fc["CatBoost"]),
        }

        c_f1, c_f2, c_f3 = st.columns(3)
        for col, name in zip(
            [c_f1, c_f2, c_f3], ["SARIMA", "XGBoost", "CatBoost"]
        ):
            mae, rmse, mape = stats_fc2[name]
            with col:
                st.metric(f"{name} MAE", f"{mae:,.0f}")
                st.metric(f"{name} RMSE", f"{rmse:,.0f}")
                st.metric(f"{name} MAPE (%)", f"{mape:,.1f}")
