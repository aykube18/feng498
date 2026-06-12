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
import matplotlib.ticker as mticker

# =============================================================================
# 1. DATA LOADING - ESKİ VE YENİ METHODLAR
# =============================================================================

def load_file(uploaded_file):
    """Eski system: CSV/Excel dosya yükleme"""
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def load_material_data_from_excel(filepath: str, material_code: int, sheet_name=None):
    """
    YENİ SYSTEM: ForecastEOQ.xlsx formatından veri yükleme
    Columns: Tarih, Mal Giriş, Tüketim Miktarı, Unit Price, Para Birimi, Lead time
    """
    try:
        if sheet_name is None:
            sheet_name = str(material_code)
        
        df = pd.read_excel(filepath, sheet_name=sheet_name)
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
        }

        return df, metadata

    except Exception as e:
        st.warning(f"Error loading material {material_code}: {str(e)}")
        return None, None

def clean_raw(df):
    """Eski system: Veri temizleme"""
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
# 2. AGGREGATION FUNCTIONS - ESKİ SYSTEM
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
# 3. FORECAST MODELS - ESKİ SYSTEM
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
# 4. ABC – XYZ ANALYSIS - ESKİ SYSTEM
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
# 5. YENİ INVENTORY SYSTEM - REAL DATA BASED
# =============================================================================

SERVICE_LEVEL = 0.95
ORDERING_COST = 2792.0
HOLDING_COST_PCT = 0.25
STOCKOUT_COST_PER_UNIT = 500.0
DEFAULT_LEAD_TIME_DAYS = 7

def calculate_real_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    YENİ: Gerçek envanter hesaplaması
    Envanter = Önceki Envanter + Giriş - Tüketim (negatif değer stok azlığını gösterir)
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
            stockout_list.append(-current_inv)
        else:
            stockout_list.append(0)
        
        inventory_list.append(current_inv)

    df["Envanter"] = inventory_list
    df["Stockout"] = stockout_list
    return df

def calculate_real_inventory_costs(df: pd.DataFrame, metadata: dict, 
                                   ordering_cost=ORDERING_COST,
                                   holding_cost_pct=HOLDING_COST_PCT,
                                   stockout_cost=STOCKOUT_COST_PER_UNIT):
    """
    YENİ: Gerçek envanter maliyetleri
    - Holding Cost: Günlük Envanter × Günlük Oran × Birim Fiyatı
    - Ordering Cost: Sipariş Sayısı × Sipariş Maliyeti
    - Stockout Cost: Stok Azlığı × Birim Başına Maliyet
    """
    df = df.copy()
    unit_price = metadata["unit_price"]
    
    daily_holding_cost_rate = holding_cost_pct / 365
    df["Holding_Cost_Daily"] = df["Envanter"].clip(lower=0) * daily_holding_cost_rate * unit_price
    df["Is_Order"] = df["Mal Giriş"] > 0
    df["Ordering_Cost_Daily"] = df["Is_Order"] * ordering_cost
    df["Stockout_Cost_Daily"] = df["Stockout"] * stockout_cost
    df["Total_Cost_Daily"] = df["Holding_Cost_Daily"] + df["Ordering_Cost_Daily"] + df["Stockout_Cost_Daily"]
    
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
        "total_holding_cost": round(total_holding_cost, 2),
        "total_ordering_cost": round(total_ordering_cost, 2),
        "total_stockout_cost": round(total_stockout_cost, 2),
        "total_cost": round(total_cost, 2),
        "avg_daily_cost": round(avg_daily_cost, 2),
        "annual_cost": round(annual_cost, 2),
        "total_orders": int(total_orders),
        "total_stockout_units": round(total_stockout_units, 2),
        "stockout_days": int(stockout_days),
    }
    
    return df, cost_summary

def calculate_statistics(df: pd.DataFrame, metadata: dict,
                        ordering_cost=ORDERING_COST,
                        holding_cost_pct=HOLDING_COST_PCT) -> dict:
    """
    YENİ: EOQ ve ROP hesaplaması
    """
    consumption = df["Tüketim Miktarı"].values
    inflow = df["Mal Giriş"].values
    inventory = df["Envanter"].values

    days = len(df)
    total_consumption = consumption.sum()
    total_inflow = inflow.sum()
    avg_daily_consumption = total_consumption / days if days > 0 else 0

    consumption_nonzero = consumption[consumption > 0]
    std_consumption = consumption_nonzero.std() if len(consumption_nonzero) > 1 else avg_daily_consumption * 0.3
    std_daily = std_consumption / np.sqrt(len(df)) if len(df) > 1 else avg_daily_consumption * 0.3

    cv = (std_daily / avg_daily_consumption * 100) if avg_daily_consumption > 0 else 0

    annual_demand = avg_daily_consumption * 365
    holding_cost = holding_cost_pct * metadata["unit_price"]

    if annual_demand > 0 and holding_cost > 0:
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    else:
        eoq = avg_daily_consumption * 30

    eoq = max(1.0, eoq)

    lead_time = metadata["lead_time"]
    lt_mean = avg_daily_consumption * lead_time
    lt_std = std_daily * math.sqrt(lead_time)

    z = norm.ppf(SERVICE_LEVEL)
    rop = max(0.0, lt_mean + z * lt_std)

    return {
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
        "annual_demand": round(annual_demand, 0),
        "lead_time": lead_time,
    }

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
# 7. VISUALIZATION FUNCTIONS - YENİ SYSTEM
# =============================================================================

def create_eoq_graph(df: pd.DataFrame, stats: dict, cost_summary: dict, metadata: dict) -> None:
    """
    YENİ: EOQ Testere Dişi Grafiği
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#F8FAFC")

    ax1.set_facecolor("#FFFFFF")
    days = range(len(df))

    ax1.fill_between(days, 0, stats["rop"], alpha=0.08, color="#16A34A", label="Safety Stock Zone")
    ax1.fill_between(days, 0, df["Envanter"], where=(df["Envanter"] < 0), 
                     alpha=0.15, color="#EF4444", label="Stockout Zone")

    ax1.plot(days, df["Envanter"], color="#DC2626", linewidth=2.5,
             label="Inventory Level", marker="o", markersize=2, alpha=0.85, zorder=3)

    ax1.axhline(y=stats["rop"], color="#16A34A", linestyle="--", linewidth=2.5,
                label=f"ROP = {stats['rop']:.0f}", zorder=2, alpha=0.8)
    ax1.axhline(y=stats["eoq"], color="#D97706", linestyle="--", linewidth=2,
                label=f"EOQ = {stats['eoq']:.0f}", zorder=1, alpha=0.6)
    ax1.axhline(y=stats["avg_inventory"], color="#7C3AED", linestyle=":", linewidth=2.5,
                label=f"Avg = {stats['avg_inventory']:.0f}", zorder=2, alpha=0.8)
    ax1.axhline(y=0, color="#000000", linestyle="-", linewidth=1.5, zorder=0, alpha=0.5)

    ax1.set_xlabel("Days", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_ylabel("Inventory (units)", fontsize=11, fontweight="bold", color="#374151")
    ax1.set_title(f"📦 EOQ Analysis: {metadata['material_code']}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    info_text = (f"📊 CONFIG\nCode: {metadata['material_code']}\n"
                f"Lead Time: {metadata['lead_time']}d\nUnit Price: {metadata['currency']}{metadata['unit_price']:.2f}\n"
                f"Avg Daily: {stats['avg_daily_consumption']:.2f}\nStd Dev: {stats['std_daily']:.2f}\n"
                f"CV: {stats['cv_percent']:.1f}%\n━━━━━━━━\nTotal Cost: {metadata['currency']}{cost_summary['total_cost']:,.0f}\n"
                f"Stockout: {cost_summary['stockout_days']}d")

    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=8,
             ha="left", va="top", fontweight="bold", color="#111827",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF3C7", alpha=0.95,
                      edgecolor="#D97706", linewidth=1.5))

    ax2.set_facecolor("#FFFFFF")
    x = np.arange(len(df))
    width = 0.35

    ax2.bar(x - width/2, df["Mal Giriş"], width, label="Inflow", color="#16A34A", alpha=0.8)
    ax2.bar(x + width/2, df["Tüketim Miktarı"], width, label="Consumption", color="#DC2626", alpha=0.8)

    ax2.set_xlabel("Days", fontsize=11, fontweight="bold", color="#374151")
    ax2.set_ylabel("Quantity", fontsize=11, fontweight="bold", color="#374151")
    ax2.set_title("Inflow vs Consumption", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax2.spines[["top", "right"]].set_visible(False)

    if len(df) > 50:
        step = len(df) // 10
        ax2.set_xticks(range(0, len(df), step))

    plt.tight_layout()
    return fig

def create_cost_breakdown_chart(cost_summary: dict, stats: dict, metadata: dict) -> None:
    """
    YENİ: Maliyet Dağılımı Grafiği
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8FAFC")

    # Pie Chart
    ax1.set_facecolor("#FFFFFF")
    costs = [
        cost_summary["total_holding_cost"],
        cost_summary["total_ordering_cost"],
        cost_summary["total_stockout_cost"]
    ]
    labels = [f"Holding\n{metadata['currency']}{costs[0]:,.0f}", 
              f"Ordering\n{metadata['currency']}{costs[1]:,.0f}",
              f"Stockout\n{metadata['currency']}{costs[2]:,.0f}"]
    colors = ["#16A34A", "#D97706", "#DC2626"]

    wedges, texts, autotexts = ax1.pie(costs, labels=labels, autopct="%1.1f%%", 
                                        colors=colors, startangle=90, textprops={"fontsize": 10, "weight": "bold"})
    ax1.set_title("Cost Breakdown", fontsize=12, fontweight="bold")

    # Summary Text
    ax2.axis("off")
    summary_text = f"""
📊 COST ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Period: {stats['total_days']} days
Total Consumption: {stats['total_consumption']:,.0f} units
Annual Demand: {stats['annual_demand']:,.0f} units
Avg Daily: {stats['avg_daily_consumption']:.2f} units

━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 COSTS (Period):
Holding: {metadata['currency']}{cost_summary['total_holding_cost']:,.2f}
Ordering: {metadata['currency']}{cost_summary['total_ordering_cost']:,.2f}
Stockout: {metadata['currency']}{cost_summary['total_stockout_cost']:,.2f}
────────────────
TOTAL: {metadata['currency']}{cost_summary['total_cost']:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 ANNUALIZED:
Avg Daily: {metadata['currency']}{cost_summary['avg_daily_cost']:.2f}
Annual: {metadata['currency']}{cost_summary['annual_cost']:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  STOCKOUT METRICS:
Total Shortage: {cost_summary['total_stockout_units']:.0f} units
Stockout Days: {cost_summary['stockout_days']} days
Orders Placed: {cost_summary['total_orders']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 RECOMMENDATIONS:
EOQ: {stats['eoq']:.0f} units
ROP: {stats['rop']:.0f} units
Avg Inventory: {stats['avg_inventory']:.0f} units
    """

    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=9,
             ha="left", va="top", family="monospace", fontweight="bold",
             bbox=dict(boxstyle="round,pad=1", facecolor="#F3F4F6", alpha=0.95,
                      edgecolor="#6B7280", linewidth=1.5))

    plt.tight_layout()
    return fig

# =============================================================================
# 8. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="📦 DSS - Integrated Inventory", layout="wide")
st.title("📦 Integrated Decision Support System - YENİ ENVANTER")

# Sidebar - Input Selection
st.sidebar.title("⚙️ Configuration")
input_mode = st.sidebar.radio("Data Source", ["Classic System (Monthly)", "New System (Real Data)"])

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIC SYSTEM - ESKİ UI
# ─────────────────────────────────────────────────────────────────────────────

if input_mode == "Classic System (Monthly)":
    st.header("1️⃣ Classical System - Monthly Data")
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

    # TAB 1 — EDA
    with tab1:
        st.header("📊 Exploratory Data Analysis")
        st.write(df.head())
        st.write("Total records:", len(df))

    monthly = get_monthly(df)
    weekly = get_weekly(df)

    if not monthly:
        st.warning("No sufficient monthly data. Check your file.")
        series_m = None
        code, desc = "", ""
    else:
        items_list = [f"{c} - {d}" for (c, d) in monthly.keys()]
        selected = st.sidebar.selectbox("Select a product", items_list)
        key = list(monthly.keys())[items_list.index(selected)]
        code, desc = key
        series_m = monthly[key]

    # TAB 2 — ABC–XYZ
    with tab2:
        st.header("🧮 ABC–XYZ Analysis")

        if not monthly:
            st.info("ABC-XYZ analysis requires sufficient monthly data.")
        else:
            df_abc = abc_analysis(monthly)
            df_xyz = xyz_analysis(monthly)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Analysis")
                st.dataframe(df_abc, use_container_width=True)

            with col2:
                st.subheader("XYZ Analysis")
                st.dataframe(df_xyz, use_container_width=True)

            st.subheader("ABC-XYZ Matrix")
            try:
                merged = df_abc.merge(df_xyz[["Material", "XYZ"]], on="Material", how="left")
                merged["ABC_XYZ"] = merged["ABC"].fillna("") + merged["XYZ"].fillna("")

                if isinstance(merged, pd.DataFrame) and not merged.empty:
                    try:
                        styled = merged.style.map(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])
                    except AttributeError:
                        styled = merged.style.applymap(color_abc_xyz, subset=["ABC", "XYZ", "ABC_XYZ"])

                    st.markdown(styled.to_html(), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # TAB 3 — FORECAST
    with tab3:
        st.header("📈 Forecasting Models")

        if series_m is None:
            st.info("No product selected.")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("ARIMA")
                fc_arima = arima_forecast(series_m)
                if fc_arima is not None:
                    st.line_chart(fc_arima)
                else:
                    st.warning("ARIMA failed.")

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

    # TAB 4 — INVENTORY
    with tab4:
        st.header("📦 Inventory Optimization")

        if series_m is None:
            st.info("No product selected.")
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

                if mean_daily <= 0:
                    st.warning("Not enough demand data.")
                else:
                    st.metric("Daily Mean", f"{mean_daily:.2f}")
                    st.metric("Daily Std Dev", f"{std_daily:.2f}")

            with col_right:
                st.info("Classic inventory visualization")

    # TAB 5 — TIME SERIES
    with tab5:
        st.header("⏳ Time Series")
        if series_m is None:
            st.info("No product selected.")
        else:
            st.line_chart(series_m)

    # TAB 6 — DASHBOARD
    with tab6:
        st.header("📋 Summary Dashboard")
        if series_m is None:
            st.info("No product selected.")

# ─────────────────────────────────────────────────────────────────────────────
# NEW SYSTEM - YENİ ENVANTER UI
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.header("2️⃣ NEW System - Real Inventory Data (ForecastEOQ Format)")
    
    uploaded_excel = st.file_uploader("Upload ForecastEOQ.xlsx", type=["xlsx"])
    
    if uploaded_excel is None:
        st.info("Please upload ForecastEOQ.xlsx file containing material sheets.")
        st.stop()
    
    # Material code input
    st.sidebar.subheader("📊 Material Selection")
    material_code = st.sidebar.number_input("Enter Material Code", value=600080, step=1)
    
    # Load data
    df_new, metadata = load_material_data_from_excel(uploaded_excel, material_code)
    
    if df_new is None or metadata is None:
        st.error(f"Could not load material {material_code}. Check Excel sheet name.")
        st.stop()
    
    # Calculate real inventory
    df_new = calculate_real_inventory(df_new)
    
    # Calculate statistics
    stats = calculate_statistics(df_new, metadata)
    
    # Calculate costs
    df_new, cost_summary = calculate_real_inventory_costs(df_new, metadata)
    
    st.success(f"✅ Material {material_code} loaded successfully!")
    
    # Create tabs
    tab_inv, tab_cost, tab_data, tab_metrics = st.tabs(
        ["📦 Inventory Graph", "💰 Cost Analysis", "📋 Data Table", "📊 Metrics"]
    )
    
    # TAB 1: Inventory Graph
    with tab_inv:
        st.header(f"📦 Inventory Level - Material {material_code}")
        
        fig = create_eoq_graph(df_new, stats, cost_summary, metadata)
        st.pyplot(fig)
        plt.close()
    
    # TAB 2: Cost Analysis
    with tab_cost:
        st.header(f"💰 Cost Analysis - Material {material_code}")
        
        fig = create_cost_breakdown_chart(cost_summary, stats, metadata)
        st.pyplot(fig)
        plt.close()
    
    # TAB 3: Data Table
    with tab_data:
        st.header("📋 Detailed Data")
        
        # Filter by date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df_new["Tarih"].min().date())
        with col2:
            end_date = st.date_input("End Date", df_new["Tarih"].max().date())
        
        df_filtered = df_new[(df_new["Tarih"].dt.date >= start_date) & (df_new["Tarih"].dt.date <= end_date)]
        
        st.dataframe(df_filtered[[
            "Tarih", "Mal Giriş", "Tüketim Miktarı", "Envanter", "Stockout",
            "Holding_Cost_Daily", "Ordering_Cost_Daily", "Stockout_Cost_Daily", "Total_Cost_Daily"
        ]].round(2), use_container_width=True)
    
    # TAB 4: Metrics
    with tab_metrics:
        st.header("📊 Key Metrics & Recommendations")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("EOQ (Optimal Order)", f"{stats['eoq']:.0f} units")
        col2.metric("ROP (Reorder Point)", f"{stats['rop']:.0f} units")
        col3.metric("Avg Inventory", f"{stats['avg_inventory']:.0f} units")
        col4.metric("Lead Time", f"{metadata['lead_time']} days")
        
        st.markdown("---")
        
        col5, col6, col7 = st.columns(3)
        col5.metric("Avg Daily Consumption", f"{stats['avg_daily_consumption']:.2f} units")
        col6.metric("Std Dev", f"{stats['std_daily']:.2f} units")
        col7.metric("CV", f"{stats['cv_percent']:.1f}%")
        
        st.markdown("---")
        
        col8, col9, col10, col11 = st.columns(4)
        col8.metric("Total Orders", f"{cost_summary['total_orders']}")
        col9.metric("Stockout Days", f"{cost_summary['stockout_days']}")
        col10.metric("Total Shortage", f"{cost_summary['total_stockout_units']:.0f} units")
        col11.metric("Service Level", f"{SERVICE_LEVEL*100:.0f}%")
        
        st.markdown("---")
        
        col12, col13, col14 = st.columns(3)
        col12.metric("Total Holding Cost", f"{metadata['currency']}{cost_summary['total_holding_cost']:,.2f}")
        col13.metric("Total Ordering Cost", f"{metadata['currency']}{cost_summary['total_ordering_cost']:,.2f}")
        col14.metric("Total Stockout Cost", f"{metadata['currency']}{cost_summary['total_stockout_cost']:,.2f}")
        
        st.markdown("---")
        col15, col16 = st.columns(2)
        col15.metric("Total Cost (Period)", f"{metadata['currency']}{cost_summary['total_cost']:,.2f}")
        col16.metric("Annualized Cost", f"{metadata['currency']}{cost_summary['annual_cost']:,.2f}")

st.sidebar.markdown("---")
st.sidebar.info(
    "🔄 **Integrated System**\n\n"
    "- **Classic**: Aylık veriler, ABC-XYZ, Forecast\n"
    "- **New**: Gerçek envanter, Maliyet analizi, Stok azlığı\n\n"
    "🎯 Service Level: 95%\n"
    "📊 Lead Time: 7 günü varsayılan"
)
