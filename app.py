import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("models/linear_model.pkl")
scaler = joblib.load("models/scaler.pkl")  # Load the saved StandardScaler

# Page title
st.title("📈 NIFTY 50 Stock Price Prediction Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a view", ["Overview", "Model Prediction", "Backtesting"])

if page == "Overview":
    st.header("📊 Project Summary")
    st.markdown("""
    - **Model**: Linear Regression  
    - **Target**: Next day closing price of NIFTY 50  
    - **R² Score**: 99.62%  
    - **Win Rate (Backtest)**: 52.89%  
    - **Sharpe Ratio**: 0.01  
    - **Total Profit**: ₹1455.35  
    """)

elif page == "Model Prediction":
    st.header("🧠 Predict Next Day Price")

    # Input fields for all 12 features
    open_ = st.number_input("Open Price", value=18000.0)
    high = st.number_input("High Price", value=18100.0)
    low = st.number_input("Low Price", value=17950.0)
    close = st.number_input("Previous Close", value=18000.0)
    lag1 = st.number_input("Lag_1", value=18000.0)
    lag3 = st.number_input("Lag_3", value=17900.0)
    lag7 = st.number_input("Lag_7", value=17800.0)
    ma20 = st.number_input("MA_20", value=17980.0)
    ma100 = st.number_input("MA_100", value=17500.0)
    bb_high = st.number_input("BB_High", value=18200.0)
    bb_low = st.number_input("BB_Low", value=17800.0)
    bb_width = st.number_input("BB_Width", value=400.0)

    # Combine inputs
    features = [[close, low, open_, high, lag1, lag3, lag7, ma100, bb_low, ma20, bb_high, bb_width]]

    if st.button("Predict"):
        # Scale input features
        scaled_features = scaler.transform(features)
        # Predict
        pred = model.predict(scaled_features)[0]
        st.success(f"📢 Predicted Next Day Closing Price: ₹{pred:.2f}")

elif page == "Backtesting":
    st.header("📈 Backtest Summary")

    st.markdown("### Cumulative Return (Simulated Trades)")

    fig, ax = plt.subplots()
    cum_returns = pd.read_csv("data/backtest_cum_return.csv")
    ax.plot(cum_returns['Cumulative_Return'], label="Cumulative Return")
    ax.set_title("Backtesting Cumulative Profit")
    ax.set_ylabel("₹ Profit")
    ax.grid(True)
    st.pyplot(fig)
