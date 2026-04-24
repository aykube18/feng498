import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import io

# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================

def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def clean_raw(df):
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

def get_monthly(df):
    result = {}
    for (code, desc), grp in df.groupby(["Malzeme", "Aciklama"]):
        m = grp.set_index("Tarih")["Miktar"].resample("MS").sum()
        if len(m) < 6:
            continue
        full = pd.date_range(start=m.index.min(), end=m.index.max(), freq="MS")
        m = m.reindex(full, fill_value=0)
        result[(code, desc)] = m
    return result

def get_weekly(df):
    result = {}
    for (code, desc), grp in df.groupby(["Malzeme", "Aciklama"]):
        w = grp.set_index("Tarih")["Miktar"].resample("W").sum()
        if len(w) < 20:
            continue
        full = pd.date_range(start=w.index.min(), end=w.index.max(), freq="W")
        w = w.reindex(full, fill_value=0)
        result[(code, desc)] = w
    return result

# =============================================================================
# 3. FORECAST MODELLERİ
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
# 4. ENVANTER OPTİMİZASYONU
# =============================================================================

def inventory_metrics(series, service=0.95, lt=7, order_cost=2000, hold_pct=0.25, unit_cost=1):
    if len(series) < 6:
        return None

    mean_m = series.mean()
    std_m = series.std()

    mean_d = mean_m / 30
    std_d = std_m / np.sqrt(30)

    if mean_d <= 0:
        return None

    z = norm.ppf(service)
    mu_L = mean_d * lt
    sigma_L = std_d * np.sqrt(lt)

    ss = max(0, z * sigma_L)
    rop = mu_L + ss

    annual = mean_d * 365
    hold_cost = hold_pct * unit_cost
    eoq = np.sqrt((2 * annual * order_cost) / hold_cost)

    return {
        "mean_monthly": mean_m,
        "std_monthly": std_m,
        "mean_daily": mean_d,
        "std_daily": std_d,
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq": eoq
    }

# =============================================================================
# 5. STREAMLIT ARAYÜZÜ
# =============================================================================

st.set_page_config(page_title="DSS", layout="wide")
st.title("📦 Çok Aşamalı Karar Destek Sistemi")

# -------------------------
# ADIM 1 — VERİ YÜKLEME
# -------------------------
st.header("1️⃣ Veri Yükleme")
uploaded = st.file_uploader("Excel (xlsx/xls) veya CSV yükleyin", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Devam etmek için bir dosya yükleyin.")
    st.stop()

df = load_file(uploaded)
df = clean_raw(df)

st.success("Veri başarıyla yüklendi!")

# -------------------------
# ADIM 2 — EDA
# -------------------------
st.header("2️⃣ EDA — Keşifsel Veri Analizi")
st.write(df.head())
st.write("Toplam kayıt:", len(df))

# -------------------------
# ADIM 3 — TIME SERIES
# -------------------------
st.header("3️⃣ Zaman Serisi Oluşturma")

monthly = get_monthly(df)
weekly = get_weekly(df)

if len(monthly) == 0:
    st.error("Aylık seri oluşturulamadı.")
    st.stop()

items = [f"{c} - {d}" for (c,d) in monthly.keys()]
selected = st.selectbox("Ürün seçin", items)

key = list(monthly.keys())[items.index(selected)]
code, desc = key
series_m = monthly[key]

st.subheader("Aylık Talep Serisi")
st.line_chart(series_m)

# -------------------------
# ADIM 4 — FORECAST
# -------------------------
st.header("4️⃣ Tahmin Modülleri")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ARIMA")
    fc_arima = arima_forecast(series_m)
    if fc_arima is not None:
        st.line_chart(fc_arima)
    else:
        st.warning("ARIMA tahmini yapılamadı.")

with col2:
    st.subheader("XGBoost")
    y_test_xgb, preds_xgb, mae_xgb, rmse_xgb = ml_forecast(series_m, "xgboost")
    if preds_xgb is not None:
        st.write(f"MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")
        st.line_chart(pd.DataFrame({"Gerçek": y_test_xgb, "Tahmin": preds_xgb}))
    else:
        st.warning("Yetersiz veri.")

with col3:
    st.subheader("CatBoost")
    y_test_cat, preds_cat, mae_cat, rmse_cat = ml_forecast(series_m, "catboost")
    if preds_cat is not None:
        st.write(f"MAE: {mae_cat:.2f}, RMSE: {rmse_cat:.2f}")
        st.line_chart(pd.DataFrame({"Gerçek": y_test_cat, "Tahmin": preds_cat}))
    else:
        st.warning("Yetersiz veri.")

# -------------------------
# ADIM 5 — ENVANTER OPTİMİZASYONU
# -------------------------
st.header("5️⃣ Envanter Optimizasyonu")

service = st.slider("Servis Seviyesi", 0.80, 0.999, 0.95)
lt = st.number_input("Teslim Süresi (gün)", 1, 60, 7)
order_cost = st.number_input("Sipariş Maliyeti", 1.0, 10000.0, 2000.0)
hold_pct = st.number_input("Stok Bulundurma Oranı", 0.01, 1.0, 0.25)
unit_cost = st.number_input("Birim Maliyet", 0.1, 1000.0, 1.0)

metrics = inventory_metrics(series_m, service, lt, order_cost, hold_pct, unit_cost)

if metrics:
    st.metric("Güvenlik Stoğu", f"{metrics['safety_stock']:.1f}")
    st.metric("ROP", f"{metrics['reorder_point']:.1f}")
    st.metric("EOQ", f"{metrics['eoq']:.1f}")
else:
    st.warning("Envanter hesapları için yeterli veri yok.")

# -------------------------
# ADIM 6 — DASHBOARD
# -------------------------
st.header("6️⃣ Dashboard — Özet")

st.subheader("Ürün Bilgisi")
st.write(f"**Kod:** {code}")
st.write(f"**Açıklama:** {desc}")

st.subheader("Aylık Talep")
st.line_chart(series_m)

st.subheader("Tahmin Sonuçları")
if fc_arima is not None:
    st.write("ARIMA Tahmini")
    st.line_chart(fc_arima)

if preds_xgb is not None:
    st.write("XGBoost Tahmini")
    st.line_chart(pd.DataFrame({"Gerçek": y_test_xgb, "Tahmin": preds_xgb}))

if preds_cat is not None:
    st.write("CatBoost Tahmini")
    st.line_chart(pd.DataFrame({"Gerçek": y_test_cat, "Tahmin": preds_cat}))

st.subheader("Envanter Metrikleri")

col1, col2, col3 = st.columns(3)
col1.metric("Aylık Ortalama", f"{metrics['mean_monthly']:.1f}")
col2.metric("Aylık Std", f"{metrics['std_monthly']:.1f}")
col3.metric("Günlük Ortalama", f"{metrics['mean_daily']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Günlük Std", f"{metrics['std_daily']:.2f}")
col5.metric("Güvenlik Stoğu", f"{metrics['safety_stock']:.1f}")
col6.metric("ROP", f"{metrics['reorder_point']:.1f}")

st.metric("EOQ", f"{metrics['eoq']:.1f}")
