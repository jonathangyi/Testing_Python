import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
file_path = "Binance_BTCUSDT_d.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path, skiprows=1)

# Rename columns for Prophet
df.columns = ["Unix", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume_BTC", "Volume_USDT", "Tradecount"]
df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime
df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})  # Prophet format

# Initialize and train Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df)

# Predict for the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Bitcoin Price Forecast (Next 30 Days)")
plt.show()

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("btc_forecast_results.csv", index=False)
print("Forecast saved to btc_forecast_results.csv")

import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(df['ds'], df['y'], label="Actual Prices", color='black', linestyle='solid')

# Plot forecast
plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Prices", color='blue', linestyle='dashed')

# Add confidence intervals
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)

# Labels and title
plt.xlabel("Date")
plt.ylabel("Bitcoin Price (USDT)")
plt.title("Actual vs Forecasted Bitcoin Prices")
plt.legend()

# Show plot
plt.show()
