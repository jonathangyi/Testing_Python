import pandas as pd
import requests
import time
import schedule
import matplotlib.pyplot as plt
from prophet import Prophet

# Function to fetch real-time BTC price data
def fetch_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=100"
    response = requests.get(url)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QAV", "NTrades", "TBBV", "TBQV", "Ignore"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp
    df["y"] = df["Close"].astype(float)  # Convert price to float
    return df[["ds", "y"]]

# Function to train Prophet and predict future BTC prices
def predict_btc():
    print("\n🔄 Fetching new BTC data...")
    df = fetch_btc_data()

    # Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Actual Prices", color="black")
    plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Prices", color="blue", linestyle="dashed")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="blue", alpha=0.2)
    plt.xlabel("Date")
    plt.ylabel("Bitcoin Price (USDT)")
    plt.title("Real-Time Bitcoin Price Prediction")
    plt.legend()
    plt.show()

    print("✅ Forecast updated!")

# Schedule to run every 1 hour (adjust as needed)
#schedule.every(1).hours.do(predict_btc)
schedule.every(30).seconds.do(predict_btc)

# Run immediately once and then every interval
predict_btc()

print("⏳ Waiting for next update... Press CTRL+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(1)
