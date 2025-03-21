import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'  # Change to a font that supports emojis

import matplotlib.dates as mdates
import ta
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Function to fetch real-time BTC price
def get_live_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url).json()
    return float(response["price"])

# Function to fetch historical BTC data
def fetch_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=365"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QAV", "NTrades", "TBBV", "TBQV", "Ignore"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp
    df["y"] = df["Close"].astype(float)  # Convert price to float
    return df[["ds", "y"]]

# Function to predict BTC prices
def predict_btc(df, days):
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast

# Function to compute technical indicators
def compute_indicators(df):
    df["SMA_10"] = ta.trend.sma_indicator(df["y"], window=10).bfill()
    df["SMA_50"] = ta.trend.sma_indicator(df["y"], window=50).bfill()  # Updated
    df["RSI"] = ta.momentum.rsi(df["y"], window=14).bfill()  # Updated
    return df

# Streamlit App UI
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Real-Time Bitcoin Prediction Dashboard")

# Live Bitcoin Price
st.subheader("💰 Live Bitcoin Price")
btc_price = get_live_btc_price()
st.write(f"📌 **Current BTC Price: ${btc_price:,.2f}**")

# Fetch historical data
st.subheader("📊 Historical Bitcoin Data (Past Year)")
df = fetch_btc_data()
df = compute_indicators(df)
st.write(df.tail(10))  # Show last 10 rows

# User input for forecast period
days = st.slider("🔮 Select Forecast Period (Days)", min_value=7, max_value=90, value=30)

# Predict BTC prices
st.subheader(f"📅 Bitcoin Price Forecast for Next {days} Days")
forecast = predict_btc(df, days)

# Merge actual & predicted data
merged_df = pd.merge(df, forecast, on="ds", how="inner")  # Align actual & predicted values

# Compute rolling accuracy (MAPE-based)
window_size = st.slider("📊 Select Rolling Window for Accuracy (Days)", min_value=7, max_value=60, value=30)
merged_df["rolling_mape"] = merged_df["y"].rolling(window_size).apply(
    lambda x: mean_absolute_percentage_error(x, forecast["yhat"].loc[x.index]) * 100 if len(x) == window_size else None
)
merged_df["rolling_accuracy"] = 100 - merged_df["rolling_mape"]  # Convert MAPE to accuracy
rolling_accuracy_df = merged_df.dropna(subset=["rolling_accuracy"])  # Remove NaN values

# 📈 Improved Prediction Graph
df["ds"] = pd.to_datetime(df["ds"])  # Ensure correct datetime format
last_30_days = df[df["ds"] >= df["ds"].max() - pd.Timedelta(days=30)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(last_30_days["ds"], last_30_days["y"], label="Actual Prices", color="black", marker="o")
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Prices", color="blue", linestyle="dashed", marker="o")
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="blue", alpha=0.2)

# Show latest model accuracy
mape = mean_absolute_percentage_error(merged_df["y"], merged_df["yhat"]) * 100
accuracy = 100 - mape
accuracy_text = f"📊 Model Accuracy: {accuracy:.2f}%"
ax.text(0.05, 0.95, accuracy_text, transform=ax.transAxes, fontsize=12, color="green", bbox=dict(facecolor="white", alpha=0.5))

ax.set_xlabel("Date")
ax.set_ylabel("Bitcoin Price (USDT)")
ax.set_title(f"Bitcoin Price Prediction for the Next {days} Days")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# 📜 Forecasted Data Table
st.subheader("📜 Forecasted Prices")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days))

# 📊 Rolling Accuracy Over Time Graph
st.subheader("📉 Rolling Model Accuracy Over Time")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rolling_accuracy_df["ds"], rolling_accuracy_df["rolling_accuracy"], label="Rolling Accuracy (%)", color="blue", linestyle="solid", marker="o")
ax.axhline(y=rolling_accuracy_df["rolling_accuracy"].mean(), color="red", linestyle="dashed", label="Avg Accuracy")
ax.set_xlabel("Date")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Rolling Accuracy Over Time (Window: {window_size} Days)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# 📊 Moving Averages Graph
st.subheader("📊 Moving Averages (SMA 10 & SMA 50)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(last_30_days["ds"], last_30_days["y"], label="Actual Prices", color="black", marker="o")
ax.plot(last_30_days["ds"], last_30_days["SMA_10"], label="10-day SMA", color="red", linestyle="dashed")
ax.plot(last_30_days["ds"], last_30_days["SMA_50"], label="50-day SMA", color="green", linestyle="dashed")
ax.set_xlabel("Date")
ax.set_ylabel("Bitcoin Price (USDT)")
ax.set_title("Bitcoin Price with Moving Averages")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# 📈 RSI Indicator with Buy/Sell Signals
st.subheader("📈 RSI Indicator (Buy/Sell Signals)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(last_30_days["ds"], last_30_days["RSI"], label="RSI", color="purple", marker="o")
ax.axhline(70, linestyle="dashed", color="red", label="Overbought (Sell)")
ax.axhline(30, linestyle="dashed", color="green", label="Oversold (Buy)")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# RSI-based Buy/Sell Alerts
latest_rsi = df["RSI"].iloc[-1]
if latest_rsi > 70:
    st.warning(f"🚨 RSI is {latest_rsi:.2f} → Overbought! Consider Selling!")
elif latest_rsi < 30:
    st.success(f"✅ RSI is {latest_rsi:.2f} → Oversold! Consider Buying!")

# Refresh Button
if st.button("🔄 Refresh Data"):
    st.rerun()
