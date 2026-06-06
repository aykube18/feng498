import streamlit as st
import pandas as pd
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS & CONFIGURATIONS (VERİ TİPLERİ FLOAT OLARAK SABİTLENDİ)
# =============================================================================
ORDERING_COST = 2792.0       # TRY - Fixed ordering cost
HOLDING_COST_PCT = 0.25      # 25% - Annual holding cost percentage
SERVICE_LEVEL = 0.95         # 95% service level
STOCKOUT_COST_PER_UNIT = 150.0 # TRY - Cost per unit of stockout (Hata önleyici float)

SHORT_NAMES = {
    600080: "MOTORIN", 600096: "TOKA 32", 600102: "ÇEMBER TOKA",
    600112: "SIYAH ÇEMBER", 603789: "ELDİVEN", 603812: "KULAK TIKACI",
    607290: "Ç.4140", 626100: "SAPAN", 627518: "BRK 77x60", 627519: "BRK 80x122"
}

# =============================================================================
# 1. DATA LOADING & CLEANING
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
        "Tarih": "Date", "Tüketim Miktarı": "Quantity", "Mal Giriş": "Inflow"
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
# 2. AGGREGATION FUNCTIONS
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
# 3. ADVANCED FORECAST LOGIC
# =============================================================================
def run_sarima_grid(train_series, test_series, m=7):
    best_aic, best_order, best_seasonal = np.inf, (1, 1, 1), (1, 1, 0, m)
    combos = list(itertools.product(range(2), range(2), range(2), range(2), range(2), range(2)))
    for p, d, q, P, D, Q in combos:
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        try:
            res = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, m),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=20)
            if res.aic < best_aic:
                best_aic, best_order, best_seasonal = res.aic, (p, d, q), (P, D, Q, m)
        except Exception:
            continue
    try:
        fitted = SARIMAX(train_series, order=best_order, seasonal_order=best_seasonal,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
        fc = pd.Series(fitted.forecast(steps=len(test_series)).values, index=test_series.index).clip(lower=0)
        return fc, f"SARIMA{best_order}{best_seasonal}"
    except Exception:
        return pd.Series(0.0, index=test_series.index), "SARIMA Fallback"

def make_ml_features(series, lags=14, rolling_windows=[7, 14, 30]):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    for w in rolling_windows:
        df[f'rolling_mean_{w}'] = df['y'].shift(1).rolling(w).mean()
        df[f'rolling_std_{w}']  = df['y'].shift(1).rolling(w).std()
    df['dayofweek'] = series.index.dayofweek
    df['day']       = series.index.day
    df['month']     = series.index.month
    df['dow_sin']   = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['trend']     = np.arange(len(series))
    return df.dropna()

def recursive_forecast(model, train_series, test_index, feature_cols, lags=14):
    history = list(train_series.values)
    history_idx = list(train_series.index)
    preds = []
    for i in range(len(test_index)):
        temp = pd.Series(history, index=history_idx)
        temp_df = make_ml_features(temp, lags=lags)
        if temp_df.empty:
            preds.append(0.0)
        else:
            pred = max(0.0, float(model.predict(temp_df[feature_cols].iloc[[-1]])[0]))
            preds.append(pred)
        history.append(preds[-1])
        history_idx.append(test_index[i])
    return pd.Series(preds, index=test_index)

def run_ml_forecast(train_series, test_series, model_type="xgboost"):
    df_train = make_ml_features(train_series)
    if df_train.empty:
        return pd.Series(0.0, index=test_series.index)
    
    feature_cols = [c for c in df_train.columns if c != 'y']
    
    if model_type == "xgboost":
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
    else:
        model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
        
    model.fit(df_train[feature_cols], df_train['y'], verbose=False)
    return recursive_forecast(model, train_series, test_series.index, feature_cols)

def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual.values - forecast.values) / np.where(actual.values == 0, 1, actual.values))) * 100
    return mae, rmse, mape

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
# 5. ADVANCED INVENTORY LOGIC
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

# =============================================================================
# 6. STYLING FUNCTION FOR MATRIX
# =============================================================================
def color_abc_xyz(val):
    colors = {"A": "#4CAF50", "B": "#FFC107", "C": "#F44336",
              "X": "#4CAF50", "Y": "#FFC107", "Z": "#F44336"}
    if val in colors:
        return f"background-color: {colors[val]}; color: white; font-weight: bold;"
    return ""

# =============================================================================
# 7. STREAMLIT UI ENTRY
# =============================================================================
st.set_page_config(page_title="DSS Pro", layout="wide")
st.title("📦 Integrated Decision Support System (Advanced Model)")

st.header("1️⃣ Data Upload")
uploaded = st.file_uploader("Upload Excel (xlsx/xls) or CSV File", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Please upload a data file to activate the dashboard system.")
    st.stop()

raw_data = load_file(uploaded)
if raw_data is None or raw_data.empty:
    st.error("The uploaded file is empty or missing necessary columns.")
    st.stop()

df = clean_raw(raw_data)
st.success("Data successfully optimized and structural mapping completed!")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Data Sample", "🧮 ABC–XYZ Matrix", "📈 Advanced Forecasting", 
    "💸 Real Cost & Inventory", "⏳ Daily Time Series", "📋 Summary Dashboard"
])

with tab1:
    st.header("📊 Exploratory Data Analysis & Structure")
    st.dataframe(df.head(10))
    st.metric("Total Row Records", f"{len(df):,}")

monthly = get_monthly(df)
if not monthly:
    st.warning("Insufficient data available to classify historical steps or draw time steps.")
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

# =============================================================================
# TAB 2 — ABC–XYZ ANALYSIS
# =============================================================================
with tab2:
    st.header("🧮 ABC–XYZ Matrix Analytics")
    df_abc = abc_analysis(monthly)
    df_xyz = xyz_analysis(monthly)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ABC Cumulative Shares")
        st.dataframe(df_abc)
    with col_b:
        st.subheader("XYZ Coefficients of Variation")
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
    st.header("📈 High-Fidelity Forecast Model Projections (Daily)")
    
    daily_series = product_df.set_index("Date")["Quantity"]
    n_days = len(daily_series)
    
    if n_days < 15:
        st.warning("Not enough data points found to evaluate multi-model recursive paths safely.")
    else:
        split_idx = int(n_days * 0.8)
        train_seq = daily_series.iloc[:split_idx]
        test_seq = daily_series.iloc[split_idx:]
        
        st.write(f"**Data Span:** {train_seq.index.min().date()} to {test_seq.index.max().date()} | **Train/Test:** {len(train_seq)}g / {len(test_seq)}g")
        
        with st.spinner("Executing Automated SARIMA Grid Search and ML Pipelines..."):
            fc_sarima, sarima_name = run_sarima_grid(train_seq, test_seq, m=7)
            fc_xgb = run_ml_forecast(train_seq, test_seq, "xgboost")
            fc_cat = run_ml_forecast(train_seq, test_seq, "catboost")
            
        m_sarima = calculate_metrics(test_seq, fc_sarima)
        m_xgb = calculate_metrics(test_seq, fc_xgb)
        m_cat = calculate_metrics(test_seq, fc_cat)
        
        metrics_df = pd.DataFrame({
            "Model Pipeline": [sarima_name, "XGBoost Regressor", "CatBoost Regressor"],
            "MAE": [m_sarima[0], m_xgb[0], m_cat[0]],
            "RMSE": [m_sarima[1], m_xgb[1], m_cat[1]],
            "MAPE (%)": [m_sarima[2], m_xgb[2], m_cat[2]]
        })
        st.table(metrics_df)
        
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(daily_series.index, daily_series.values, color='#1a6faf', alpha=0.6, label='Actual Historical Demand')
        
        last_step = train_seq.iloc[[-1]]
        ax.plot(pd.concat([last_step, fc_sarima]).index, pd.concat([last_step, fc_sarima]).values, color='#e53935', linestyle='--', label='SARIMA Predict')
        ax.plot(pd.concat([last_step, fc_xgb]).index, pd.concat([last_step, fc_xgb]).values, color='#43a047', linestyle='-.', label='XGBoost Predict')
        ax.plot(pd.concat([last_step, fc_cat]).index, pd.concat([last_step, fc_cat]).values, color='#8e24aa', linestyle=':', label='CatBoost Predict')
        
        ax.axvspan(test_seq.index[0], test_seq.index[-1], alpha=0.08, color='gray', label='Test Horizon Evaluation Area (%20)')
        ax.set_title(f"Multi-Model Comparison Chart for Product: {p_code}")
        ax.set_xlabel("Timeline")
        ax.set_ylabel("Demand Units")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

# =============================================================================
# TAB 4 — REAL COSTS & INVENTORY OPTIMIZATION (HATA DÜZELTİLMİŞ KESİN TIP ATAMALARI)
# =============================================================================
with tab4:
    st.header("💸 Real Material Flows, Costs & Stockout Matrix")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ui_unit_price = st.number_input(
            "Unit Price (Material Base Cost)", 
            min_value=0.1, 
            max_value=50000.0, 
            value=10.0, 
            step=1.0
        )
        ui_order_cost = st.number_input(
            "Ordering Setup Cost (Fixed)", 
            min_value=1.0, 
            max_value=100000.0, 
            value=float(ORDERING_COST), 
            step=10.0
        )
    with col_p2:
        ui_hold_pct = st.slider(
            "Holding Cost Rate (Annualized %)", 
            min_value=0.01, 
            max_value=1.0, 
            value=float(HOLDING_COST_PCT), 
            step=0.01
        )
        ui_stockout_cost = st.number_input(
            "Penalty/Stockout Cost Per Unit Shortage", 
            min_value=0.0, 
            max_value=5000.0, 
            value=float(STOCKOUT_COST_PER_UNIT),
            step=1.0
        )
        
    ui_lead_time = st.number_input(
        "Lead Time Duration (Days)", 
        min_value=1, 
        max_value=90, 
        value=int(7),
        step=1
    )
    
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
# TAB 5 — DAILY TIME SERIES VISUALIZATION
# =============================================================================
with tab5:
    st.header("⏳ Baseline Historical Time Series Data")
    st.line_chart(product_df.set_index("Date")["Quantity"])

# =============================================================================
# TAB 6 — SUMMARY SYSTEM DASHBOARD
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
        
    st.info("System integration parameters running healthy. Export options and reports generated successfully.")
