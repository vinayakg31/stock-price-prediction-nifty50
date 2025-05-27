import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("models/linear_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Create logo + title using st.columns for Streamlit compatibility
logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image("gncipl_logo.jpg", width=60)
with title_col:
    st.markdown("## 📈 NIFTY 50 Stock Price Prediction Dashboard")

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

    open_ = st.number_input("Open Price", value=18010.0)
    high = st.number_input("High Price", value=18100.0)
    low = st.number_input("Low Price", value=17900.0)
    close = st.number_input("Close", value=18000.0)
    lag1 = st.number_input("Lag_1", value=17950.0)
    lag3 = st.number_input("Lag_3 (Close 3 days ago)", value=17850.0)
    lag7 = st.number_input("Lag_7 (Close 7 days ago)", value=17700.0)
    ma20 = st.number_input("MA_20 (20-day moving average)", value=17980.0)
    ma100 = st.number_input("MA_100 (100-day moving average)", value=17500.0)
    bb_high = st.number_input("BB_High (Upper Bollinger Band)", value=18200.0)
    bb_low = st.number_input("BB_Low (Lower Bollinger Band)", value=17800.0)
    bb_width = st.number_input("BB_Width (Band Width)", value=400.0)

    features = [[close, low, open_, high, lag1, lag3, lag7, ma100, bb_low, ma20, bb_high, bb_width]]

    if st.button("Predict"):
        scaled_features = scaler.transform(features)
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
