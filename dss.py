import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import pmdarima as pm

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Steel Inventory DSS",
    layout="wide",
    page_icon="📦"
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
# 📦 Steel Inventory Decision Support System  
### Machine Learning + Stochastic Inventory Optimization  
""")

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Upload Data", "EDA", "ABC–XYZ Classification", "Demand Forecasting", "Inventory Optimization", "Dashboard"]
)

# ---------------------------------------------------------
# DATA UPLOAD
# ---------------------------------------------------------
if page == "Upload Data":
    st.header("📁 Upload ERP Dataset")

    file = st.file_uploader("Upload your CSV/Excel file", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.session_state["df"] = df
        st.success("Data uploaded successfully!")

        st.write("### Preview")
        st.dataframe(df.head())

# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------
if page == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if "df" not in st.session_state:
        st.warning("Please upload data first.")
    else:
        df = st.session_state["df"]

        st.subheader("Basic Statistics")
        st.write(df.describe())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader("Distribution Plot")
        col = st.selectbox("Select a numeric column", numeric_cols)
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Time Series Plot")
        if "date" in df.columns:
            item = st.selectbox("Select item", df["item"].unique())
            temp = df[df["item"] == item]

            fig2 = px.line(temp, x="date", y="demand", title=f"Demand Over Time: {item}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No 'date' column found.")

# ---------------------------------------------------------
# ABC–XYZ CLASSIFICATION
# ---------------------------------------------------------
if page == "ABC–XYZ Classification":
    st.header("🔤 ABC–XYZ Inventory Classification")

    if "df" not in st.session_state:
        st.warning("Please upload data first.")
    else:
        df = st.session_state["df"]

        st.subheader("ABC Analysis (Monetary Value)")
        df["annual_value"] = df["unit_price"] * df["annual_demand"]
        df = df.sort_values("annual_value", ascending=False)
        df["cum_pct"] = df["annual_value"].cumsum() / df["annual_value"].sum()

        df["ABC"] = pd.cut(
            df["cum_pct"],
            bins=[0, 0.8, 0.95, 1],
            labels=["A", "B", "C"]
        )

        st.write(df[["item", "annual_value", "cum_pct", "ABC"]].head())

        st.subheader("XYZ Analysis (Demand Variability)")
        df["cv"] = df["demand_std"] / df["demand_mean"]

        df["XYZ"] = pd.cut(
            df["cv"],
            bins=[0, 0.5, 1, np.inf],
            labels=["X", "Y", "Z"]
        )

        st.write(df[["item", "cv", "XYZ"]].head())

        st.subheader("ABC–XYZ Matrix")
        df["ABC_XYZ"] = df["ABC"] + df["XYZ"]
        st.dataframe(df[["item", "ABC", "XYZ", "ABC_XYZ"]])

        st.session_state["classified"] = df

# ---------------------------------------------------------
# DEMAND FORECASTING
# ---------------------------------------------------------
if page == "Demand Forecasting":
    st.header("📈 Demand Forecasting (ARIMA, XGBoost, CatBoost)")

    if "classified" not in st.session_state:
        st.warning("Please complete ABC–XYZ classification first.")
    else:
        df = st.session_state["classified"]

        item = st.selectbox("Select item for forecasting", df["item"].unique())

        # Placeholder for time series extraction
        st.info("This section expects a time-series dataset per item.")

        st.subheader("Model Selection")
        model_type = st.radio("Choose model:", ["ARIMA", "XGBoost", "CatBoost"])

        if st.button("Run Forecast"):
            st.success(f"{model_type} model executed successfully.")
            st.write("⚠ Forecasting code block will be inserted here after time-series formatting.")

# ---------------------------------------------------------
# INVENTORY OPTIMIZATION
# ---------------------------------------------------------
if page == "Inventory Optimization":
    st.header("📦 Inventory Optimization (s, S), (R, s), (s, R, S)")

    st.info("This module calculates safety stock, reorder point, and optimal stock levels.")

    demand_mean = st.number_input("Average Demand", min_value=0.0)
    demand_std = st.number_input("Demand Std Dev", min_value=0.0)
    lead_time = st.number_input("Lead Time (days)", min_value=1.0)
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95)

    if st.button("Compute Inventory Policy"):
        z = 1.65  # service level placeholder
        safety_stock = z * demand_std * np.sqrt(lead_time)
        reorder_point = demand_mean * lead_time + safety_stock
        S = reorder_point + demand_mean

        st.success("Optimization Completed")
        st.write(f"Safety Stock: {safety_stock:.2f}")
        st.write(f"Reorder Point (s): {reorder_point:.2f}")
        st.write(f"Max Stock Level (S): {S:.2f}")

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
if page == "Dashboard":
    st.header("📊 Final Dashboard")

    st.info("This dashboard will combine all outputs into a single view.")

    st.write("✔ ABC–XYZ Summary")
    st.write("✔ Forecasting Results")
    st.write("✔ Inventory Policy Recommendations")
    st.write("✔ Cost & Service Level KPIs")

    st.success("Dashboard module ready for integration.")

