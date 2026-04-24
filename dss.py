# app.py
# Streamlit DSS: EDA → Time Series → Forecast → Inventory Optimization → Dashboard

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

# ML & Time Series
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Streamlit
import streamlit as st

# =============================================================================
# 1. VERİ OKUMA
# =============================================================================

def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name="RawDataCombined")

    df = df.rename(columns={
        "Malzeme": "Malzeme",
        "Malzeme kısa metni": "Aciklama",
        "Hareket türleri metni": "HareketTuru",
        "Kayıt tarihi": "Tarih",
        "Miktar Abs": "Miktar",
        "Temel ölçü birimi": "Birim",
        "WhichDepo?": "Depo"
    })

    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")
    df["Miktar"] = pd.to_numeric(df["Miktar"], errors="coerce").fillna(0)
    df["Malzeme"] = df["Malzeme"].astype(str)
    df = df.dropna(subset=["Tarih"])

    return df


# =============================================================================
# 2. AGGREGATE FONKSİYONLARI
# =============================================================================

def get_monthly_series(df: pd.DataFrame):
    result = {}
    for (code, desc), grp in df.groupby(["Malzeme", "Aciklama"]):
        monthly = grp.set_index("Tarih")["Miktar"].resample("MS").sum()
        if len(monthly) < 6:
            continue
        full_index = pd.date_range(start=monthly.index.min(),
                                   end=monthly.index.max(), freq="MS")
        monthly = monthly.reindex(full_index, fill_value=0)
        result[(code, desc)] = monthly
    return result


def get_weekly_series(df: pd.DataFrame):
    result = {}
    for (code, desc), grp in df.groupby(["Malzeme", "Aciklama"]):
        weekly = grp.set_index("Tarih")["Miktar"].resample("W").sum()
        if len(weekly) < 20:
            continue
        full_index = pd.date_range(start=weekly.index.min(),
                                   end=weekly.index.max(), freq="W")
        weekly = weekly.reindex(full_index, fill_value=0)
        result[(code, desc)] = weekly
    return result


# =============================================================================
# 3. ARIMA FORECAST
# =============================================================================

def arima_forecast(series: pd.Series, horizon: int = 6):
    try:
        model = ARIMA(series, order=(1,1,1))
        res = model.fit()
        fc = res.forecast(steps=horizon)
        return fc
    except:
        return None


# =============================================================================
# 4. ML FEATURE ENGINEERING
# =============================================================================

def make_features(series: pd.Series, lags: int = 12):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series))
    df = df.dropna()
    return df


def ml_forecast(series: pd.Series, model_type="xgboost"):
    df = make_features(series)
    if df.empty or len(df) < 20:
        return None, None, None, None

    X = df.drop("y", axis=1)
    y = df["y"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(X_test) == 0:
        return None, None, None, None

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            random_seed=42,
            verbose=0
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return y_test, preds, mae, rmse


# =============================================================================
# 5. KRİTİK STOK HESAPLARI
# =============================================================================

def calculate_inventory_metrics(series_monthly: pd.Series,
                                service_level: float = 0.95,
                                lead_time_days: int = 7,
                                ordering_cost: float = 2000.0,
                                holding_cost_pct: float = 0.25,
                                unit_cost: float = 1.0):

    if len(series_monthly) < 6:
        return None

    mean_monthly = series_monthly.mean()
    std_monthly = series_monthly.std(ddof=1)

    mean_daily = mean_monthly / 30
    std_daily = std_monthly / np.sqrt(30) if std_monthly > 0 else 0

    if mean_daily <= 0:
        return None

    from scipy.stats import norm
    z = norm.ppf(service_level)

    mu_L = mean_daily * lead_time_days
    sigma_L = std_daily * np.sqrt(lead_time_days)

    safety_stock = max(0, z * sigma_L)
    reorder_point = max(0, mu_L + safety_stock)

    annual_demand = mean_daily * 365
    holding_cost = holding_cost_pct * unit_cost
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

    return {
        "mean_monthly": mean_monthly,
        "std_monthly": std_monthly,
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "eoq": eoq
    }


# =============================================================================
# 6. STREAMLIT ARAYÜZÜ (SEKMELİ)
# =============================================================================

def main():
    st.set_page_config(page_title="DSS", layout="wide")

    st.title("📦 Entegre Karar Destek Sistemi")
    st.markdown("Veri → EDA → Zaman Serisi → Tahmin → Envanter Optimizasyonu → Dashboard")

    # --- Dosya seçimi ---
    default_file = "23_march_updated_AYBUKE_UI icinbunukullan.xlsx"
    filepath = st.sidebar.text_input("Excel dosya adı", value=default_file)

    if not os.path.exists(filepath):
        st.error("Excel dosyası bulunamadı.")
        st.stop()

    df = load_raw_data(filepath)
    monthly_dict = get_monthly_series(df)
    weekly_dict = get_weekly_series(df)

    items = [f"{code} - {desc}" for (code, desc) in monthly_dict.keys()]
    selected_item = st.sidebar.selectbox("Ürün seçiniz", items)

    selected_key = list(monthly_dict.keys())[items.index(selected_item)]
    code, desc = selected_key
    series_monthly = monthly_dict[selected_key]

    # --- Sekmeler ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 EDA", "⏳ Time Series", "📈 Forecast", "📦 Envanter", "📋 Dashboard"]
    )

    # =========================================================================
    # TAB 1 — EDA
    # =========================================================================
    with tab1:
        st.header("📊 Keşifsel Veri Analizi (EDA)")
        st.write(df.head())
        st.write("Aylık talep serisi:")
        st.line_chart(series_monthly)

    # =========================================================================
    # TAB 2 — TIME SERIES
    # =========================================================================
    with tab2:
        st.header("⏳ Zaman Serisi İncelemesi")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(series_monthly.index, series_monthly.values)
        ax.set_title(f"{code} — Aylık Talep")
        ax.grid(True)
        st.pyplot(fig)

    # =========================================================================
    # TAB 3 — FORECAST
    # =========================================================================
    with tab3:
        st.header("📈 Tahmin Modülleri")

        st.subheader("ARIMA Tahmini")
        fc_arima = arima_forecast(series_monthly)
        if fc_arima is not None:
            st.line_chart(fc_arima)
        else:
            st.warning("ARIMA tahmini yapılamadı.")

        st.subheader("XGBoost Tahmini")
        y_test_xgb, preds_xgb, mae_xgb, rmse_xgb = ml_forecast(series_monthly, "xgboost")
        if preds_xgb is not None:
            st.write(f"MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
            st.line_chart(pd.DataFrame({"Gerçek": y_test_xgb, "Tahmin": preds_xgb}))
        else:
            st.warning("XGBoost için yeterli veri yok.")

        st.subheader("CatBoost Tahmini")
        y_test_cat, preds_cat, mae_cat, rmse_cat = ml_forecast(series_monthly, "catboost")
        if preds_cat is not None:
            st.write(f"MAE: {mae_cat:.2f}, RMSE: {rmse_cat:.2f}")
            st.line_chart(pd.DataFrame({"Gerçek": y_test_cat, "Tahmin": preds_cat}))
        else:
            st.warning("CatBoost için yeterli veri yok.")

    # =========================================================================
    # TAB 4 — ENVANTER OPTİMİZASYONU
    # =========================================================================
    with tab4:
        st.header("📦 Envanter Optimizasyonu")

        service_level = st.slider("Servis Seviyesi", 0.80, 0.999, 0.95)
        lead_time = st.number_input("Teslim Süresi (gün)", 1, 60, 7)
        ordering_cost = st.number_input("Sipariş Maliyeti", 1.0, 10000.0, 2000.0)
        holding_cost_pct = st.number_input("Stok Bulundurma Oranı", 0.01, 1.0, 0.25)
        unit_cost = st.number_input("Birim Maliyet", 0.1, 1000.0, 1.0)

        metrics = calculate_inventory_metrics(
            series_monthly,
            service_level=service_level,
            lead_time_days=lead_time,
            ordering_cost=ordering_cost,
            holding_cost_pct=holding_cost_pct,
            unit_cost=unit_cost
        )

        if metrics:
            st.metric("Güvenlik Stoğu", f"{metrics['safety_stock']:.1f}")
            st.metric("ROP", f"{metrics['reorder_point']:.1f}")
            st.metric("EOQ", f"{metrics['eoq']:.1f}")
        else:
            st.warning("Envanter hesapları için yeterli veri yok.")

    # =========================================================================
    # TAB 5 — DASHBOARD
    # =========================================================================
    with tab5:
        st.header("📋 Dashboard — Özet Görünüm")

        st.subheader("Ürün Bilgisi")
        st.write(f"**Kod:** {code}")
        st.write(f"**Açıklama:** {desc}")

        st.subheader("Aylık Talep")
        st.line_chart(series_monthly)

        st.subheader("Tahmin Sonuçları")
        if fc_arima is not None:
            st.write("ARIMA Tahmini:")
            st.line_chart(fc_arima)

        if preds_xgb is not None:
            st.write("XGBoost Tahmini:")
            st.line_chart(pd.DataFrame({"Gerçek": y_test_xgb, "Tahmin": preds_xgb}))

        if preds_cat is not None:
            st.write("CatBoost Tahmini:")
            st.line_chart(pd.DataFrame({"Gerçek": y_test_cat, "Tahmin": preds_cat}))

        st.subheader("Envanter Metrikleri")
        if metrics:
            st.write(metrics)
        else:
            st.warning("Envanter metrikleri hesaplanamadı.")


if __name__ == "__main__":
    main()
