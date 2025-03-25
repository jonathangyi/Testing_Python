import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import threading
import ta
import yfinance as yf
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Function to fetch BTC live price
def get_live_btc_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return float(response.json()["price"])
    except requests.RequestException as e:
        st.error(f"Error fetching BTC price: {e}")
        return None

# Function to fetch BTC historical data
@st.cache_data(ttl=3600)
def fetch_btc_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=365"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json(), columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QAV", "NTrades", "TBBV", "TBQV", "Ignore"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["y"] = df["Close"].astype(float)
        return df[["ds", "y", "Volume"]]
    except requests.RequestException as e:
        st.error(f"Error fetching BTC data: {e}")
        return pd.DataFrame(columns=["ds", "y", "Volume"])

# Function to fetch S&P 500 stock data
@st.cache_data(ttl=3600)
def fetch_stock_data():
    try:
        spy = yf.download("SPY", period="1y")
        spy.reset_index(inplace=True)
        spy.rename(columns={"Date": "ds", "Close": "stock_price"}, inplace=True)
        return spy[["ds", "stock_price"]]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(columns=["ds", "stock_price"])

# Function to compute BTC-SPY correlation
def calculate_btc_spy_correlation(btc_df, stock_df):
    merged_df = pd.merge(btc_df, stock_df, on="ds", how="inner")
    correlation = merged_df["y"].pct_change().corr(merged_df["stock_price"].pct_change())
    return correlation, merged_df

# Streamlit UI Setup
st.set_page_config(page_title="Bitcoin Dashboard", layout="wide")
st.title("📈 Bitcoin Prediction Dashboard")

# Sidebar BTC-SPY Correlation
btc_df, stock_df = fetch_btc_data(), fetch_stock_data()
correlation, merged_df = calculate_btc_spy_correlation(btc_df, stock_df)
st.sidebar.metric("🔗 BTC-SPY Correlation", f"{correlation:.2f}")

# Live BTC Price Update
price_placeholder = st.empty()

def update_price_live():
    last_price = None
    while True:
        current_price = get_live_btc_price()
        if current_price and current_price != last_price:
            color = "green" if (last_price and current_price > last_price) else "red"
            price_placeholder.markdown(f"### 💰 Live Bitcoin Price: <span style='color:{color};'>${current_price:,.2f}</span>", unsafe_allow_html=True)
            last_price = current_price
        time.sleep(1)

if "price_thread" not in st.session_state:
    st.session_state.price_thread = threading.Thread(target=update_price_live, daemon=True)
    st.session_state.price_thread.start()

# Display BTC & Stock Data
tab1, tab2 = st.tabs(["Market Overview", "Correlation Analysis"])

with tab1:
    st.subheader("📊 BTC & SPY Market Data")
    if not btc_df.empty and not stock_df.empty:
        st.dataframe(btc_df.tail(10))
        st.dataframe(stock_df.tail(10))
    else:
        st.error("Failed to fetch market data.")

with tab2:
    st.subheader("📉 BTC-SPY Correlation")
    st.write(f"Bitcoin and S&P 500 have a correlation of **{correlation:.2f}** over the past year.")
    st.line_chart(merged_df.set_index("ds")["y"])
    st.line_chart(merged_df.set_index("ds")["stock_price"])
    st.write("Correlation is calculated using the percentage change in prices.")
    st.write("Correlation value ranges from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.")
    st.write("A positive correlation means that when one asset goes up, the other asset also goes up, and vice versa for a negative correlation.")
    st.write("Correlation does not imply causation.")
    