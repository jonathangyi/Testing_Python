import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import ta
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import threading
import plotly.graph_objects as go
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources (only need to run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch real-time BTC price
def get_live_btc_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return float(response.json()["price"])
    except (requests.RequestException, KeyError, ValueError) as e:
        st.error(f"Error fetching BTC price: {e}")
        return None

# Function to fetch historical BTC data
@st.cache_data(ttl=3600)
def fetch_btc_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=365"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QAV", "NTrades", "TBBV", "TBQV", "Ignore"])
        df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["y"] = df["Close"].astype(float)
        return df[["ds", "y", "Volume"]]
    except (requests.RequestException, ValueError, KeyError) as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame(columns=["ds", "y", "Volume"])

# Function to fetch stock market data (S&P 500 as proxy)
@st.cache_data(ttl=3600)
def fetch_stock_data():
    try:
        # Fetch S&P 500 data
        spy = yf.download("SPY", period="1y")
        spy.reset_index(inplace=True)
        spy.rename(columns={"Date": "ds", "Close": "stock_price"}, inplace=True)
        return spy[["ds", "stock_price"]]
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(columns=["ds", "stock_price"])

# Function to fetch relevant news and calculate sentiment
@st.cache_data(ttl=21600)  # Cache for 6 hours
def fetch_news_sentiment():
    try:
        # List of crypto news sources
        news_sources = [
            "https://www.coindesk.com/",
            "https://cointelegraph.com/",
            "https://decrypt.co/",
            "https://www.theblockcrypto.com/"
        ]
        
        articles = []
        sentiments = []
        
        for source in news_sources[:2]:  # Limit to first 2 sources to speed up processing
            try:
                article = Article(source)
                article.download()
                article.parse()
                
                if article.text:
                    articles.append({
                        'source': source,
                        'title': article.title,
                        'text': article.text[:500] + "..."  # Truncate for display
                    })
                    
                    # Calculate sentiment
                    sentiment = sia.polarity_scores(article.text)
                    sentiments.append(sentiment['compound'])
            except Exception as e:
                st.warning(f"Could not fetch news from {source}: {e}")
                continue
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return articles, avg_sentiment
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return [], 0

# Function to get major world events (simplified mock implementation)
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_world_events():
    # In a production app, you would integrate with a news API or events database
    # This is just a placeholder example
    events = [
        {"date": "2025-03-15", "event": "Fed increased interest rates by 0.25%", "impact": -0.2},
        {"date": "2025-03-10", "event": "Major tech company announced Bitcoin holdings", "impact": 0.3},
        {"date": "2025-03-05", "event": "New crypto regulations announced in EU", "impact": -0.1},
        {"date": "2025-02-28", "event": "Major institutional investment in crypto announced", "impact": 0.4}
    ]
    return pd.DataFrame(events)

# Enhanced prediction function incorporating external factors
def predict_btc_enhanced(btc_df, stock_df, sentiment_score, days):
    try:
        # Join datasets on date
        merged_df = pd.merge(btc_df, stock_df, on="ds", how="left")
        
        # Fill missing values
        merged_df = merged_df.fillna(method='ffill')
        
        # Add stock correlation features
        merged_df["stock_change"] = merged_df["stock_price"].pct_change()
        merged_df["btc_change"] = merged_df["y"].pct_change()
        
        # Calculate rolling correlation
        merged_df["correlation"] = merged_df["btc_change"].rolling(window=30).corr(merged_df["stock_change"])
        
        # Add day of week feature (market cyclicality)
        merged_df["day_of_week"] = merged_df["ds"].dt.dayofweek
        
        # Traditional Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        
        # Add stock price as regressor
        model.add_regressor('stock_price')
        
        # Add correlation as regressor
        model.add_regressor('correlation')
        
        # Clean and prepare data for Prophet
        prophet_df = merged_df.dropna().copy()
        
        # Fit the model
        model.fit(prophet_df[["ds", "y", "stock_price", "correlation"]])
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days)
        
        # Add stock price forecast to future dataframe
        # In a real implementation, you would have a separate model to predict stock prices
        # Here we simply extend the last value
        last_stock_price = merged_df["stock_price"].iloc[-1]
        future["stock_price"] = last_stock_price
        
        # Add correlation to future dataframe
        last_correlation = merged_df["correlation"].iloc[-1]
        future["correlation"] = last_correlation
        
        # Make prediction
        forecast = model.predict(future)
        
        # Adjust forecast based on latest sentiment (simple linear adjustment)
        # Scale sentiment to reasonable percentage impact
        sentiment_impact = sentiment_score * 0.05  # 5% max impact
        forecast["yhat"] = forecast["yhat"] * (1 + sentiment_impact)
        forecast["yhat_lower"] = forecast["yhat_lower"] * (1 + sentiment_impact)
        forecast["yhat_upper"] = forecast["yhat_upper"] * (1 + sentiment_impact)
        
        return forecast, merged_df
    except Exception as e:
        st.error(f"Error in enhanced prediction: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to compute technical indicators
def compute_indicators(df):
    if df.empty:
        return df
    
    # Technical indicators
    df["SMA_10"] = ta.trend.sma_indicator(df["y"], window=10).bfill()
    df["SMA_50"] = ta.trend.sma_indicator(df["y"], window=50).bfill()
    df["RSI"] = ta.momentum.rsi(df["y"], window=14).bfill()
    
    # Add Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.volatility.bollinger_hband(df["y"]), ta.volatility.bollinger_mavg(df["y"]), ta.volatility.bollinger_lband(df["y"])
    
    # Add MACD
    df["macd"] = ta.trend.macd(df["y"])
    df["macd_signal"] = ta.trend.macd_signal(df["y"])
    df["macd_diff"] = ta.trend.macd_diff(df["y"])
    
    # Volume indicators
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].astype(float)
        df["volume_ema"] = ta.volume.volume_ema_indicator(df["Volume"], df["y"])
    
    return df

# Train and evaluate a machine learning model as complement to Prophet
def train_ml_model(df, forecast_days):
    if df.empty or len(df) < 60:  # Need enough data
        return None, None
    
    try:
        # Feature engineering
        df_ml = df.copy()
        
        # Create time-based features
        df_ml["day_of_week"] = df_ml["ds"].dt.dayofweek
        df_ml["month"] = df_ml["ds"].dt.month
        df_ml["quarter"] = df_ml["ds"].dt.quarter
        
        # Add lag features (previous n days)
        for lag in [1, 3, 7, 14]:
            df_ml[f"lag_{lag}"] = df_ml["y"].shift(lag)
            
        # Add rolling statistics
        for window in [7, 14, 30]:
            df_ml[f"rolling_mean_{window}"] = df_ml["y"].rolling(window=window).mean()
            df_ml[f"rolling_std_{window}"] = df_ml["y"].rolling(window=window).std()
        
        # Drop missing values
        df_ml = df_ml.dropna()
        
        # Create target variable - predict price n days in future
        prediction_days = min(forecast_days, 14)  # Limit to 14 days for ML model
        df_ml[f"target_{prediction_days}d"] = df_ml["y"].shift(-prediction_days)
        
        # Remove rows with missing target
        df_ml = df_ml.dropna(subset=[f"target_{prediction_days}d"])
        
        # Split into features and target
        X = df_ml.drop(["ds", "y", f"target_{prediction_days}d"], axis=1)
        y = df_ml[f"target_{prediction_days}d"]
        
        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rf_accuracy = 100 - mape
        
        # Prepare for prediction
        last_row = df_ml.iloc[-1:].copy()
        ml_predictions = []
        
        # Generate features for future days
        for i in range(1, forecast_days + 1):
            # Update date
            new_row = last_row.copy()
            new_date = last_row["ds"].iloc[0] + pd.Timedelta(days=i)
            new_row["ds"] = new_date
            new_row["day_of_week"] = new_date.dayofweek
            new_row["month"] = new_date.month
            new_row["quarter"] = new_date.quarter
            
            # Make prediction for this row
            X_pred = new_row.drop(["ds", "y", f"target_{prediction_days}d"], axis=1)
            pred = model.predict(X_pred)[0]
            
            ml_predictions.append({
                "ds": new_date,
                "ml_prediction": pred
            })
            
            # Update last row for next iteration
            last_row = new_row
            last_row["y"] = pred
            for lag in [1, 3, 7, 14]:
                if i >= lag:
                    idx = i - lag
                    last_row[f"lag_{lag}"] = ml_predictions[idx]["ml_prediction"] if idx < len(ml_predictions) else last_row["y"].iloc[0]
        
        ml_pred_df = pd.DataFrame(ml_predictions)
        
        return ml_pred_df, rf_accuracy
    except Exception as e:
        st.error(f"Error in ML model: {e}")
        return None, None

# Streamlit App UI
st.set_page_config(page_title="Advanced Bitcoin Prediction", layout="wide")

st.title("📈 Advanced Bitcoin Prediction Dashboard with World Events & Stock Market Integration")

# Create tabs for organization
tab1, tab2, tab3 = st.tabs(["Main Dashboard", "News & World Events", "Technical Analysis"])

# Create a placeholder for the live price
price_placeholder = st.empty()

# Function to continuously update the price
def update_price_in_background():
    last_price = None
    while True:
        current_price = get_live_btc_price()
        if current_price and current_price != last_price:
            # Determine price change direction for color coding
            if last_price is not None:
                if current_price > last_price:
                    price_color = "green"
                    price_change = "▲"
                elif current_price < last_price:
                    price_color = "red"
                    price_change = "▼"
                else:
                    price_color = "gray"
                    price_change = "•"
                
                # Update the placeholder with new price and color
                price_placeholder.markdown(
                    f"""
                    ### 💰 Live Bitcoin Price
                    <div style="display: flex; align-items: center;">
                        <h3 style="margin: 0; color: {price_color};">
                            ${current_price:,.2f} {price_change}
                        </h3>
                        <span style="margin-left: 10px; color: {price_color}; font-size: 0.8em;">
                            {abs(current_price - last_price):,.2f} ({abs(current_price - last_price) / last_price * 100:.2f}%)
                        </span>
                    </div>
                    <div style="font-size: 0.8em; color: gray;">Last updated: {time.strftime("%H:%M:%S")}</div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                # First time showing the price
                price_placeholder.markdown(
                    f"""
                    ### 💰 Live Bitcoin Price
                    <h3 style="margin: 0;">${current_price:,.2f}</h3>
                    <div style="font-size: 0.8em; color: gray;">Last updated: {time.strftime("%H:%M:%S")}</div>
                    """, 
                    unsafe_allow_html=True
                )
            
            last_price = current_price
        
        time.sleep(1)  # Update every second

# Initialize the price update thread
if 'price_thread' not in st.session_state:
    st.session_state.price_thread = threading.Thread(target=update_price_in_background)
    st.session_state.price_thread.daemon = True
    st.session_state.price_thread.start()

# Main dashboard tab
with tab1:
    # Fetch data
    btc_df = fetch_btc_data()
    stock_df = fetch_stock_data()
    articles, sentiment_score = fetch_news_sentiment()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🌍 Global Market Context")
        
        # Stock market correlation
        if not stock_df.empty and not btc_df.empty:
            merged = pd.merge(btc_df, stock_df, on="ds", how="inner")
            if len(merged) > 30:
                btc_returns = merged["y"].pct_change()
                spy_returns = merged["stock_price"].pct_change()
                correlation = btc_returns.corr(spy_returns)
                
                st.metric(
                    "BTC-SPY Correlation", 
                    f"{correlation:.2f}", 
                    delta=f"{(correlation - btc_returns[-30:-1].corr(spy_returns[-30:-1])):.2f}",
                    delta_color="normal"
                )
                
                # Simple display of SPY performance
                spy_last = stock_df["stock_price"].iloc[-1]
                spy_prev = stock_df["stock_price"].iloc[-2]
                spy_change = (spy_last - spy_prev) / spy_prev * 100
                
                st.metric(
                    "S&P 500 (SPY)", 
                    f"${spy_last:.2f}", 
                    delta=f"{spy_change:.2f}%",
                    delta_color="normal"
                )
        
        # News sentiment indicator
        sentiment_label = "Bullish" if sentiment_score > 0.05 else "Bearish" if sentiment_score < -0.05 else "Neutral"
        sentiment_color = "green" if sentiment_score > 0.05 else "red" if sentiment_score < -0.05 else "gray"
        
        st.metric(
            "News Sentiment", 
            sentiment_label, 
            delta=f"{sentiment_score:.2f}",
            delta_color="normal"
        )
        
        world_events = get_world_events()
        st.write("Recent Impactful Events:")
        for _, event in world_events.iterrows():
            impact_color = "green" if event["impact"] > 0 else "red"
            st.markdown(f"**{event['date']}**: {event['event']} <span style='color:{impact_color};'>({event['impact']:+.1f})</span>", unsafe_allow_html=True)

    with col1:
        if not btc_df.empty:
            # User input for forecast period
            days = st.slider("🔮 Select Forecast Period (Days)", min_value=7, max_value=90, value=30)
            
            # Compute technical indicators
            btc_df = compute_indicators(btc_df)
            
            # Run enhanced prediction
            forecast, enhanced_df = predict_btc_enhanced(btc_df, stock_df, sentiment_score, days)
            
            # Get ML prediction as complementary model
            ml_predictions, ml_accuracy = train_ml_model(btc_df, days)
            
            if not forecast.empty: