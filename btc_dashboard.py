import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Function to fetch real-time BTC price
def get_live_btc_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return float(response.json()["price"])
    except (requests.RequestException, KeyError, ValueError) as e:
        st.error(f"Error fetching BTC price: {e}")
        return None

# Function to fetch historical BTC data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_btc_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=365"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QAV", "NTrades", "TBBV", "TBQV", "Ignore"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp
        df["y"] = df["Close"].astype(float)  # Convert price to float
        return df[["ds", "y"]]
    except (requests.RequestException, ValueError, KeyError) as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame(columns=["ds", "y"])

# Function to predict BTC prices
def predict_btc(df, days):
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        return forecast
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return pd.DataFrame()

# Function to compute technical indicators
def compute_indicators(df):
    if df.empty:
        return df
    
    # Fixed the deprecated fillna(method='bfill') warnings
    df["SMA_10"] = ta.trend.sma_indicator(df["y"], window=10).bfill()
    df["SMA_50"] = ta.trend.sma_indicator(df["y"], window=50).bfill()
    df["RSI"] = ta.momentum.rsi(df["y"], window=14).bfill()
    return df

# Streamlit App UI
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("📈 Real-Time Bitcoin Prediction Dashboard")

# Live Bitcoin Price
st.subheader("💰 Live Bitcoin Price")
btc_price = get_live_btc_price()
if btc_price:
    st.write(f"📌 **Current BTC Price: ${btc_price:,.2f}**")

# Fetch historical data
st.subheader("📊 Historical Bitcoin Data (Past Year)")
df = fetch_btc_data()

if not df.empty:
    df = compute_indicators(df)
    st.write(df.tail(10))  # Show last 10 rows

    # User input for forecast period
    days = st.slider("🔮 Select Forecast Period (Days)", min_value=7, max_value=90, value=30)

    # Predict BTC prices
    st.subheader(f"📅 Bitcoin Price Forecast for Next {days} Days")
    forecast = predict_btc(df, days)

    if not forecast.empty:
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

        # Using Streamlit's native chart capabilities instead of matplotlib
        st.subheader("📈 Bitcoin Price Prediction")
        
        # Create prediction dataframe for Streamlit chart
        chart_data = pd.DataFrame({
            "Date": forecast["ds"],
            "Predicted Price": forecast["yhat"],
            "Lower Bound": forecast["yhat_lower"],
            "Upper Bound": forecast["yhat_upper"]
        })
        
        # Add actual prices for the last 30 days
        actual_data = pd.DataFrame({
            "Date": last_30_days["ds"],
            "Actual Price": last_30_days["y"]
        })
        
        # Display accuracy info
        mape = mean_absolute_percentage_error(merged_df["y"], merged_df["yhat"]) * 100
        accuracy = 100 - mape
        st.info(f"📊 Model Accuracy: {accuracy:.2f}%")
        
        # Use Streamlit's line chart
        st.line_chart(
            data=pd.concat([
                actual_data.set_index("Date")["Actual Price"],
                chart_data.set_index("Date")[["Predicted Price", "Lower Bound", "Upper Bound"]]
            ], axis=1)
        )
        
        # 📜 Forecasted Data Table
        st.subheader("📜 Forecasted Prices")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days))
        
        # Display rolling accuracy using Streamlit's native charts
        st.subheader("📉 Rolling Model Accuracy Over Time")
        accuracy_chart = pd.DataFrame({
            "Date": rolling_accuracy_df["ds"],
            "Rolling Accuracy (%)": rolling_accuracy_df["rolling_accuracy"],
            "Average Accuracy": [rolling_accuracy_df["rolling_accuracy"].mean()] * len(rolling_accuracy_df)
        }).set_index("Date")
        
        st.line_chart(accuracy_chart)
        
        # Moving Averages using Streamlit's native charts
        st.subheader("📊 Moving Averages (SMA 10 & SMA 50)")
        ma_chart = pd.DataFrame({
            "Date": last_30_days["ds"],
            "Bitcoin Price": last_30_days["y"],
            "10-day SMA": last_30_days["SMA_10"],
            "50-day SMA": last_30_days["SMA_50"]
        }).set_index("Date")
        
        st.line_chart(ma_chart)
        
        # RSI Indicator with custom HTML for better visualization
        st.subheader("📈 RSI Indicator (Buy/Sell Signals)")
        
        rsi_chart = pd.DataFrame({
            "Date": last_30_days["ds"],
            "RSI": last_30_days["RSI"]
        }).set_index("Date")
        
        st.line_chart(rsi_chart)
        
        # Add visual horizontal lines for RSI thresholds
        st.markdown("""
        <style>
        .rsi-info {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .overbought {
            background-color: rgba(255, 0, 0, 0.1);
            border-left: 5px solid red;
        }
        .neutral {
            background-color: rgba(128, 128, 128, 0.1);
            border-left: 5px solid gray;
        }
        .oversold {
            background-color: rgba(0, 128, 0, 0.1);
            border-left: 5px solid green;
        }
        </style>
        
        <div class="rsi-info overbought">
            <strong>Overbought (Sell Signal):</strong> RSI > 70
        </div>
        <div class="rsi-info neutral">
            <strong>Neutral:</strong> 30 ≤ RSI ≤ 70
        </div>
        <div class="rsi-info oversold">
            <strong>Oversold (Buy Signal):</strong> RSI < 30
        </div>
        """, unsafe_allow_html=True)
        
        # RSI-based Buy/Sell Alerts
        latest_rsi = df["RSI"].iloc[-1]
        if latest_rsi > 70:
            st.warning(f"🚨 RSI is {latest_rsi:.2f} → Overbought! Consider Selling!")
        elif latest_rsi < 30:
            st.success(f"✅ RSI is {latest_rsi:.2f} → Oversold! Consider Buying!")
        else:
            st.info(f"ℹ️ RSI is {latest_rsi:.2f} → Neutral")
else:
    st.error("Failed to fetch historical data. Please check your internet connection and try again.")

# Refresh Button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()  # Clear cached data
    st.rerun()