# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = r"C:\Time_Series_Forecast\Data_Sets\Microsoft_Stock.csv"
MODEL_PATH = r"C:\Time_Series_Forecast\ARIMA_model.pkl"


def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df

df = load_data(DATA_PATH)

# --- Load ARIMA model ---
@st.cache_resource
def load_model(path):
    return joblib.load(path)

arima_model = load_model(MODEL_PATH)

st.title("Microsoft Stock Price Forecasting")
st.write("Visualize historical stock prices and forecast future prices using ARIMA.")

if st.checkbox("Show raw data"):
    st.dataframe(df.head(20))

st.subheader("Historical Close Price")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["Close"], color="royalblue", linewidth=2, marker="o", markersize=4, markerfacecolor="orange")
ax.set_title("Stock Closing Price Over Time", fontsize=16, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("ARIMA Forecast")
steps = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

if st.button("Forecast"):
    model_fit = arima_model.fit()
    forecast = model_fit.forecast(steps=int(steps))

    st.write(f"Forecast for next {steps} days:")
    forecast_df = pd.DataFrame(forecast, columns=["Forecast"])
    st.dataframe(forecast_df)

    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(df.index, df["Close"], label="Historical", color="blue")
    ax2.plot(pd.date_range(df.index[-1], periods=steps+1, freq='B')[1:], forecast, label="Forecast", color="green")
    ax2.set_title("ARIMA Forecast of Stock Close Price", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    st.pyplot(fig2)