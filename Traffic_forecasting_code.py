# Smart City Traffic Forecasting - Internship Project (UCT)

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset (replace with your CSV file path)
# df = pd.read_csv('traffic_data.csv')

# For demonstration purposes, we create sample data
df = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    'y': [i + (i%7)*10 for i in range(100)]
})

# Forecast using Prophet
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot results
model.plot(forecast)
plt.title("Traffic Forecast for Smart City Junctions")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.tight_layout()
plt.show()
