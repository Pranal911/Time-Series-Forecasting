import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = "Data_Sets/Microsoft_Stock.csv" 
MODEL_PATH = "ARIMA_model.pkl" 
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file was not found at the expected path: {path}. Check your GitHub repository structure.")
        return pd.DataFrame() 

df = load_data(DATA_PATH)

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Error: The model file was not found at: {path}. Please ensure it is committed to GitHub.")
        return None 

arima_model = load_model(MODEL_PATH)

st.title("Microsoft Stock Price Forecasting")
st.write("Visualize historical stock prices and forecast future prices using ARIMA.")

if df.empty:
    st.stop()

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
    if arima_model is None:
        st.warning("Cannot run forecast because the model failed to load.")
    else:
        try:
            with st.spinner('Calculating forecast...'):
                model_fit = arima_model.fit()
                forecast = model_fit.forecast(steps=int(steps))
        
            st.write(f"Forecast for next {steps} days:")
            forecast_df = pd.DataFrame(forecast, columns=["Forecast"])
            st.dataframe(forecast_df)
        
            last_date = df.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq='B')[1:]
        
            fig2, ax2 = plt.subplots(figsize=(12,6))
            ax2.plot(df.index, df["Close"], label="Historical", color="blue")
            ax2.plot(forecast_dates, forecast, label="Forecast", color="green")
            ax2.set_title("ARIMA Forecast of Stock Close Price", fontsize=16, fontweight="bold")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price (USD)")
            ax2.legend()
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")