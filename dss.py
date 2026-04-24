# app.py
# Tam entegre: Veri okuma + Forecast + Kritik stok + Streamlit dashboard

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

# Zaman serisi ve ML
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Görselleştirme
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Streamlit
import streamlit as st

# ==============================
# 1. VERİ OKUMA
# ==============================

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


# ==============================
# 2. AGGREGATE FONKSİYONLARI
# ==============================

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


# ==============================
# 3. BASİT ARIMA FORECAST
# ==============================

def arima_forecast(series: pd.Series, forecast_horizon: int = 6):
    # Çok ağır grid search yerine sabit küçük bir model: ARIMA(1,1,1)
    try:
        model = ARIMA(series, order=(1, 1, 1))
        res = model.fit()
        fc = res.forecast(steps=forecast_horizon)
        return fc, res
    except Exception:
        return None, None


# ==============================
# 4. ML FEATURE ENGINEERING
# ==============================

def make_features(series: pd.Series, lags: int = 12):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series))
    df = df.dropna()
    return df


def ml_forecast(series: pd.Series, model_type: str = "xgboost"):
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


# ==============================
# 5. KRİTİK STOK HESAPLARI
# ==============================

def calculate_inventory_metrics(series_monthly: pd.Series,
                                service_level: float = 0.95,
                                lead_time_days: int = 7,
                                ordering_cost: float = 2000.0,
                                holding_cost_pct: float = 0.25,
                                unit_cost: float = 1.0):
    """
    Basit (Q,R) politikası:
    - Ortalama günlük talep = aylık ort / 30
    - Günlük std = aylık std / sqrt(30)
    - Güvenlik stoğu = z * sigma_L
    - ROP = mu_L + SS
    - EOQ = sqrt(2DS / H)
    """
    if len(series_monthly) < 6:
        return None

    mean_monthly = series_monthly.mean()
    std_monthly = series_monthly.std(ddof=1)

    mean_daily = mean_monthly / 30.0
    std_daily = std_monthly / np.sqrt(30.0) if std_monthly > 0 else 0.0

    # Talep yoksa anlamsız
    if mean_daily <= 0:
        return None

    # Normal dağılım varsayımı
    from scipy.stats import norm
    z = norm.ppf(service_level)

    mu_L = mean_daily * lead_time_days
    sigma_L = std_daily * np.sqrt(lead_time_days)

    safety_stock = max(0.0, z * sigma_L)
    reorder_point = max(0.0, mu_L + safety_stock)

    annual_demand = mean_daily * 365
    holding_cost = holding_cost_pct * unit_cost
    if holding_cost <= 0:
        eoq = mean_daily * 30
    else:
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


# ==============================
# 6. GÖRSEL FONKSİYONLAR
# ==============================

def plot_time_series_with_forecast(series: pd.Series,
                                   fc_arima: pd.Series | None,
                                   y_test_xgb, preds_xgb,
                                   y_test_cat, preds_cat,
                                   title: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # 1) Orijinal + ARIMA forecast
    ax = axes[0]
    ax.plot(series.index, series.values, label="Gerçek", color="#2563EB")
    if fc_arima is not None:
        ax.plot(fc_arima.index, fc_arima.values, label="ARIMA Tahmin", color="#DC2626")
    ax.set_title("Aylık Talep ve ARIMA Tahmini")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2) XGBoost
    ax = axes[1]
    if y_test_xgb is not None and preds_xgb is not None:
        ax.plot(y_test_xgb.index, y_test_xgb.values, label="Gerçek", color="#2563EB")
        ax.plot(y_test_xgb.index, preds_xgb, label="XGBoost Tahmin", color="#16A34A")
        ax.set_title("XGBoost Test Dönemi Tahmini")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "XGBoost için yeterli veri yok", ha="center", va="center")
        ax.axis("off")

    # 3) CatBoost
    ax = axes[2]
    if y_test_cat is not None and preds_cat is not None:
        ax.plot(y_test_cat.index, y_test_cat.values, label="Gerçek", color="#2563EB")
        ax.plot(y_test_cat.index, preds_cat, label="CatBoost Tahmin", color="#D97706")
        ax.set_title("CatBoost Test Dönemi Tahmini")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "CatBoost için yeterli veri yok", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    return fig


# ==============================
# 7. STREAMLIT ARAYÜZÜ
# ==============================

def main():
    st.set_page_config(page_title="Talep Tahmini ve Kritik Stok DSS",
                       layout="wide")

    st.title("📦 Talep Tahmini ve Kritik Stok Karar Destek Sistemi")
    st.markdown(
        "Bu arayüz, **RawDataCombined** verisinden ürün bazlı talep serilerini çıkarır, "
        "ARIMA + XGBoost + CatBoost ile tahmin yapar ve kritik stok metriklerini hesaplar."
    )

    # --- Dosya seçimi ---
    st.sidebar.header("Veri Kaynağı")
    default_file = "23_march_updated_AYBUKE_UI icinbunukullan.xlsx"
    filepath = st.sidebar.text_input("Excel dosya adı / yolu", value=default_file)

    service_level = st.sidebar.slider("Servis Seviyesi (kritik stok için)", 0.80, 0.999, 0.95, 0.01)
    lead_time_days = st.sidebar.number_input("Teslim Süresi (gün)", min_value=1, max_value=60, value=7)
    ordering_cost = st.sidebar.number_input("Sipariş Maliyeti", min_value=1.0, value=2000.0, step=100.0)
    holding_cost_pct = st.sidebar.number_input("Yıllık Stok Bulundurma Oranı", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
    unit_cost = st.sidebar.number_input("Birim Maliyet (varsayılan)", min_value=0.1, value=1.0, step=0.1)

    if not os.path.exists(filepath):
        st.error(f"Excel dosyası bulunamadı: {filepath}")
        st.stop()

    # --- Veri yükleme ---
    df = load_raw_data(filepath)

    # --- Aylık & Haftalık seriler ---
    monthly_dict = get_monthly_series(df)
    weekly_dict = get_weekly_series(df)  # Şimdilik sadece ileride kullanmak için

    if len(monthly_dict) == 0:
        st.error("Aylık seri oluşturulabilecek yeterli veri bulunamadı.")
        st.stop()

    # Ürün seçimi
    items = [f"{code} - {desc}" for (code, desc) in monthly_dict.keys()]
    selected_item = st.selectbox("Ürün seçiniz", items)

    # Seçilen ürünün kodu & açıklaması
    selected_key = list(monthly_dict.keys())[items.index(selected_item)]
    code, desc = selected_key
    series_monthly = monthly_dict[selected_key]

    st.subheader(f"Seçilen Ürün: {code} — {desc}")

    # --- Forecastler ---
    with st.spinner("ARIMA tahmini çalıştırılıyor..."):
        fc_arima, _ = arima_forecast(series_monthly, forecast_horizon=6)

    with st.spinner("XGBoost tahmini çalıştırılıyor..."):
        y_test_xgb, preds_xgb, mae_xgb, rmse_xgb = ml_forecast(series_monthly, "xgboost")

    with st.spinner("CatBoost tahmini çalıştırılıyor..."):
        y_test_cat, preds_cat, mae_cat, rmse_cat = ml_forecast(series_monthly, "catboost")

    # --- Kritik stok metrikleri ---
    metrics = calculate_inventory_metrics(
        series_monthly,
        service_level=service_level,
        lead_time_days=lead_time_days,
        ordering_cost=ordering_cost,
        holding_cost_pct=holding_cost_pct,
        unit_cost=unit_cost
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = plot_time_series_with_forecast(
            series_monthly,
            fc_arima,
            y_test_xgb, preds_xgb,
            y_test_cat, preds_cat,
            title=f"{code} — {desc}"
        )
        st.pyplot(fig)

    with col2:
        st.markdown("### 📊 Kritik Stok ve Talep Özeti")

        if metrics is None:
            st.warning("Kritik stok hesapları için yeterli veri yok.")
        else:
            st.metric("Aylık Ortalama Talep", f"{metrics['mean_monthly']:.1f}")
            st.metric("Aylık Std. Sapma", f"{metrics['std_monthly']:.1f}")
            st.metric("Günlük Ortalama Talep", f"{metrics['mean_daily']:.2f}")
            st.metric("Günlük Std. Sapma", f"{metrics['std_daily']:.2f}")
            st.metric("Güvenlik Stoğu", f"{metrics['safety_stock']:.1f}")
            st.metric("Yeniden Sipariş Noktası (ROP)", f"{metrics['reorder_point']:.1f}")
            st.metric("Ekonomik Sipariş Miktarı (EOQ)", f"{metrics['eoq']:.1f}")

        st.markdown("---")
        st.markdown("### 🔍 Model Performansları")

        if y_test_xgb is not None:
            st.write(f"**XGBoost** — MAE: `{mae_xgb:.2f}`, RMSE: `{rmse_xgb:.2f}`")
        else:
            st.write("XGBoost için yeterli veri yok.")

        if y_test_cat is not None:
            st.write(f"**CatBoost** — MAE: `{mae_cat:.2f}`, RMSE: `{rmse_cat:.2f}`")
        else:
            st.write("CatBoost için yeterli veri yok.")

    st.markdown("---")
    st.markdown(
        "Bu dashboard, **sınırlı veriyle çalışan, hızlı ve anlaşılır bir prototip** olarak tasarlandı. "
        "İstersen model yapılarını, parametreleri ve metrikleri daha da detaylandırabiliriz."
    )


if __name__ == "__main__":
    main()
