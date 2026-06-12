# app.py

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from scipy import stats
from typing import Dict, List, Tuple
import math

import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATE_COL = "Tarih"
VALUE_COL = "Tüketim"

TEST_START = "2025-04-02"  # first predicted day (prediction starts here)

SEASONAL = 7  # weekly seasonality for daily data
DEFAULT_ORDER = (1, 1, 2)  # AICc-selected on daily training data
DEFAULT_SORDER = (1, 1, 1, SEASONAL)
ALPHA = 0.05  # -> 95% one-step prediction interval
AUTO_SELECT = False  # True -> re-run the AICc grid search each run

# Feature engineering settings (XGBoost / CatBoost)
N_LAGS = 14
ROLL_WINDOWS = (7, 14, 30)

# Inventory configuration
ORDERING_COST = 2792  # TRY - Fixed ordering cost per order
HOLDING_COST_PCT = 0.25  # 25% - Annual holding cost percentage
SERVICE_LEVEL = 0.95  # 95% service level
STOCKOUT_COST_PER_UNIT = 500  # TRY - Cost per unit of stockout (lost sales + penalty)

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

COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#6366F1",
]

# ══════════════════════════════════════════════════════════════════════════════
# FORECAST ENGINE FUNCTIONS (from forecast_eren_11_06)
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
                    best = {"score": score, "order": (p, d, q), "sorder": (P, D, Q, SEASONAL)}
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

def build_comparison(test, sarima_pred, lo, hi, xgb_pred, cat_pred) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": test["date"].values,
        "t": test["t"].values,
        "Actual": test[VALUE_COL].values.astype(float),
        "SARIMA": sarima_pred,
        "SARIMA_Lo95": lo,
        "SARIMA_Hi95": hi,
        "XGBoost": xgb_pred,
        "CatBoost": cat_pred,
    })
    return out

def plot_results(frame, train, comp, order, sorder, stats):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    boundary = train["date"].iloc[-1]

    # Top: full series + all three model predictions
    ax1.plot(frame["date"], frame[VALUE_COL], color="#1f77b4",
             linewidth=0.7, alpha=0.55, label="Actual (daily)")
    ax1.plot(comp["date"], comp["SARIMA"], color="#2ca02c",
             linewidth=1.3, label="SARIMA")
    ax1.plot(comp["date"], comp["XGBoost"], color="#d62728",
             linewidth=1.3, linestyle="--", label="XGBoost")
    ax1.plot(comp["date"], comp["CatBoost"], color="#9467bd",
             linewidth=1.3, linestyle="-.", label="CatBoost")

    ax1.axvline(boundary, color="grey", linestyle=":", linewidth=1.3)
    ax1.text(boundary, ax1.get_ylim()[1], " train | test",
             color="grey", va="top", ha="left", fontsize=9)

    ax1.set_title("Full daily series — three models", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Daily consumption")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Bottom: test-window zoom
    ax2.fill_between(
        comp["date"], comp["SARIMA_Lo95"], comp["SARIMA_Hi95"],
        color="#2ca02c", alpha=0.10, label="SARIMA 95% interval"
    )
    ax2.plot(comp["date"], comp["Actual"], color="#1f77b4",
             linewidth=1.0, marker="o", markersize=2.5, label="Actual (daily)")
    ax2.plot(comp["date"], comp["SARIMA"], color="#2ca02c",
             linewidth=1.6, label="SARIMA")
    ax2.plot(comp["date"], comp["XGBoost"], color="#d62728",
             linewidth=1.6, linestyle="--", label="XGBoost")
    ax2.plot(comp["date"], comp["CatBoost"], color="#9467bd",
             linewidth=1.6, linestyle="-.", label="CatBoost")

    pstart = pd.Timestamp(comp["date"].iloc[0]).strftime("%Y-%m-%d")
    ax2.set_title(
        f"Prediction window from {pstart}: real vs rolling predictions (daily)",
        fontsize=11, fontweight="bold",
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
        fontsize=12, fontweight="bold",
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# INVENTORY FUNCTIONS (NEW VERSION WITH STOCKOUT + COSTS)
# ══════════════════════════════════════════════════════════════════════════════

def load_material_data_from_file(file, material_code: int) -> Tuple[pd.DataFrame, Dict]:
    try:
        df = pd.read_excel(file, sheet_name=str(material_code))
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
        st.error(f"Error reading material code {material_code} → {str(e)}")
        return None, None

def calculate_real_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Envanter"] = 0.0
    df["Stockout"] = 0.0

    current_inv = 0.0
    inventory_list = []
    stockout_list = []

    for idx, row in df.iterrows():
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
        "avg_order_qty": round(
            df[df["Mal Giriş"] > 0]["Mal Giriş"].mean(), 2
        ) if total_orders > 0 else 0,
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

def create_eoq_graph(df: pd.DataFrame, stats: Dict, cost_summary: Dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#F8FAFC")

    ax1.set_facecolor("#FFFFFF")
    days = range(len(df))

    ax1.fill_between(days, 0, stats["rop"], alpha=0.08, color="#16A34A",
                     label="Safety Stock Zone")

    ax1.fill_between(
        days, 0, df["Envanter"],
        where=(df["Envanter"] < 0),
        alpha=0.15, color="#EF4444", label="Stockout Zone"
    )

    ax1.plot(days, df["Envanter"], color="#DC2626", linewidth=2.5,
             label="Real Inventory Level", marker="o", markersize=2,
             alpha=0.85, zorder=3)

    ax1.axhline(y=stats["rop"], color="#16A34A", linestyle="--", linewidth=2.5,
                label=f"Reorder Point (ROP) = {stats['rop']:.0f}", zorder=2, alpha=0.8)

    ax1.axhline(y=stats["eoq"], color="#D97706", linestyle="--", linewidth=2,
                label=f"Order Quantity (EOQ) = {stats['eoq']:.0f}", zorder=1, alpha=0.6)

    ax1.axhline(y=stats["avg_inventory"], color="#7C3AED", linestyle=":", linewidth=2.5,
                label=f"Average Inventory = {stats['avg_inventory']:.0f}", zorder=2, alpha=0.8)

    ax1.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5, zorder=0, alpha=0.5)

    ax1.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_ylabel("Inventory (units)", fontsize=12, fontweight="bold", color="#374151")
    ax1.set_title(
        f"📦 EOQ SAW-TOOTH GRAPH: {stats['material_name']}\nReal Inflow & Consumption Data",
        fontsize=13, fontweight="bold", color="#111827", pad=15,
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
        0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
        ha="left", va="top", fontweight="bold", color="#111827",
        bbox=dict(
            boxstyle="round,pad=0.8", facecolor="#FEF3C7", alpha=0.95,
            edgecolor="#D97706", linewidth=2,
        ),
    )

    ax2.set_facecolor("#FFFFFF")

    x = np.arange(len(df))
    width = 0.35

    ax2.bar(x - width/2, df["Mal Giriş"], width, label="Inflow",
            color="#16A34A", alpha=0.8, edgecolor="white")
    ax2.bar(x + width/2, df["Tüketim Miktarı"], width, label="Consumption",
            color="#DC2626", alpha=0.8, edgecolor="white")

    ax2.set_xlabel("Days", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_ylabel("Quantity (units)", fontsize=12, fontweight="bold", color="#374151")
    ax2.set_title("Daily Inflow vs Consumption", fontsize=12, fontweight="bold",
                  color="#111827", pad=15)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    if len(df) > 50:
        step = len(df) // 10
        ax2.set_xticks(range(0, len(df), step))

    plt.tight_layout()
    return fig

def create_eoq_cost_graph(stats: Dict, cost_summary: Dict):
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

    ax1.plot(Q_range, ordering_cost, linewidth=2.5, color="#DC2626",
             label="Ordering Cost", linestyle="--", alpha=0.8)
    ax1.plot(Q_range, holding_cost, linewidth=2.5, color="#16A34A",
             label="Holding Cost", linestyle="--", alpha=0.8)
    ax1.plot(Q_range, total_cost, linewidth=3.5, color="#7C3AED",
             label="Total Cost", zorder=3)

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
        0.05, 0.98, cost_text, transform=ax2.transAxes, fontsize=8,
        ha="left", va="top", family="monospace", color="#111827",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=1", facecolor="#FEF3C7", alpha=0.95,
            edgecolor="#D97706", linewidth=2,
        ),
    )

    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP LAYOUT (DSS)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Integrated Decision Support System", layout="wide")

    # Title
    st.markdown("## 🧠 Integrated Decision Support System")
    st.markdown("### 1️⃣ Data Upload")

    # File upload (main DSS file, e.g., ForecastEOQ.xlsx or transactional data)
    uploaded_file = st.file_uploader(
        "Upload Excel file (e.g., ForecastEOQ.xlsx or demand data)",
        type=["xlsx", "xls"],
        key="main_upload",
    )

    # Sidebar: material selection
    st.sidebar.markdown("### 🎯 Select a product")
    material_code = st.sidebar.selectbox(
        "Material code",
        options=MATERIAL_CODES,
        format_func=lambda x: f"{x} - {SHORT_NAMES.get(x, str(x))}",
    )

    # Tabs under main title (layout like your screenshot)
    tabs = st.tabs(
        [
            "📊 ABC/XYZ",
            "🔮 Forecast",
            "📦 Inventory",
            "⏳ Time Series",
            "📈 Summary Dashboard",
        ]
    )

    # ─────────────────────────────────────────────────────────────
    # TAB 1: ABC/XYZ (placeholder – you can plug your own logic)
    # ─────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 📊 ABC/XYZ Classification")
        st.info("ABC/XYZ module placeholder. You can integrate your existing ABC–XYZ logic here.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: FORECAST (design like your second image)
    # ─────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 🔮 Forecast Engine")

        st.markdown(
            "> Automated Model Selector: SARIMA, XGBoost, and CatBoost are compared on rolling one-step forecasts."
        )

        forecast_file = st.file_uploader(
            "Upload demand file for forecasting (columns: Tarih, Tüketim)",
            type=["xlsx", "xls"],
            key="forecast_upload",
        )

        if forecast_file is not None:
            df_forecast = pd.read_excel(forecast_file)
            if DATE_COL not in df_forecast.columns or VALUE_COL not in df_forecast.columns:
                st.error(f"File must contain columns '{DATE_COL}' and '{VALUE_COL}'.")
            else:
                frame = load_daily_from_df(df_forecast)
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

                stats_dict = {
                    "SARIMA": error_stats(comp["Actual"], comp["SARIMA"]),
                    "XGBoost": error_stats(comp["Actual"], comp["XGBoost"]),
                    "CatBoost": error_stats(comp["Actual"], comp["CatBoost"]),
                }

                # Metrics table (like top table in your screenshot)
                st.markdown("#### 📋 Model Performance (Test Set)")
                perf_df = pd.DataFrame(
                    {
                        "Model": ["SARIMA", "XGBoost Regressor", "CatBoost Regressor"],
                        "MAE": [
                            stats_dict["SARIMA"][0],
                            stats_dict["XGBoost"][0],
                            stats_dict["CatBoost"][0],
                        ],
                        "RMSE": [
                            stats_dict["SARIMA"][1],
                            stats_dict["XGBoost"][1],
                            stats_dict["CatBoost"][1],
                        ],
                        "MAPE (%)": [
                            stats_dict["SARIMA"][2],
                            stats_dict["XGBoost"][2],
                            stats_dict["CatBoost"][2],
                        ],
                    }
                )
                st.dataframe(perf_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.2f}"}))

                # Two-panel forecast plots (like your second image)
                st.markdown("#### 📈 Forecast Comparison Charts")
                fig_forecast = plot_results(frame, train, comp, order, sorder, stats_dict)
                st.pyplot(fig_forecast)

        else:
            st.info("Upload a demand file to run the forecast engine.")

    # ─────────────────────────────────────────────────────────────
    # TAB 3: INVENTORY (EOQ + Stockout + Costs)
    # ─────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📦 Inventory Optimization")

        if uploaded_file is None:
            st.info("Upload your ForecastEOQ-style Excel file to run inventory analysis.")
        else:
            df, metadata = load_material_data_from_file(uploaded_file, material_code)
            if df is None or metadata is None:
                st.error("Could not load material sheet from the uploaded file.")
            else:
                st.markdown(f"**Selected Material:** `{metadata['material_code']} - {metadata['material_name']}`")

                df_inv = calculate_real_inventory(df)
                df_cost, cost_summary = calculate_real_inventory_costs(df_inv, metadata)
                stats = calculate_statistics(df_inv, metadata)

                # EOQ saw-tooth graph with stockout
                st.markdown("#### 📉 EOQ Saw-Tooth Graph (with Stockout)")
                fig_eoq = create_eoq_graph(df_inv, stats, cost_summary)
                st.pyplot(fig_eoq)

                # EOQ cost tradeoff graph
                st.markdown("#### 💰 EOQ Cost Tradeoff")
                fig_cost = create_eoq_cost_graph(stats, cost_summary)
                st.pyplot(fig_cost)

                # Summary metrics
                st.markdown("#### 📋 Inventory & Cost Summary")
                summary_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Annual Demand (units)",
                            "EOQ (units)",
                            "ROP (units)",
                            "Average Inventory (units)",
                            "Total Holding Cost (TRY)",
                            "Total Ordering Cost (TRY)",
                            "Total Stockout Cost (TRY)",
                            "Total Annual Cost (TRY)",
                            "Stockout Days",
                            "Stockout Units",
                        ],
                        "Value": [
                            stats["annual_demand"],
                            stats["eoq"],
                            stats["rop"],
                            stats["avg_inventory"],
                            cost_summary["total_holding_cost"],
                            cost_summary["total_ordering_cost"],
                            cost_summary["total_stockout_cost"],
                            cost_summary["annual_cost"],
                            cost_summary["stockout_days"],
                            cost_summary["total_stockout_units"],
                        ],
                    }
                )
                st.dataframe(summary_df)

    # ─────────────────────────────────────────────────────────────
    # TAB 4: TIME SERIES (placeholder)
    # ─────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### ⏳ Time Series Exploration")
        st.info("Time Series module placeholder. You can plug your detailed TS analysis here.")

    # ─────────────────────────────────────────────────────────────
    # TAB 5: SUMMARY DASHBOARD (placeholder)
    # ─────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 📈 Summary Dashboard")
        st.info("Summary dashboard placeholder. Combine KPIs from Forecast + Inventory + ABC/XYZ here.")

if __name__ == "__main__":
    main()
