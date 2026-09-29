import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# 1. Load dataset
data = pd.read_csv("sales_data.csv")


# 2. Convert Date column
data["Date"] = pd.to_datetime(data["Date"])


# 3. Calculate monthly sales
monthly_sales = data.groupby(
    data["Date"].dt.to_period("M")
)["Sales"].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()


# 4. Split data
# Last 12 months = testing data
train = monthly_sales[:-12]
test = monthly_sales[-12:]


# 5. Create SARIMA model
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)


# 6. Train the model
model_fit = model.fit(disp=False)


# 7. Forecast next 12 months
forecast = model_fit.forecast(steps=12)


# 8. Calculate MAE
mae = mean_absolute_error(test, forecast)


# 9. Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))


# 10. Calculate MAPE
mape = mean_absolute_percentage_error(test, forecast) * 100


# 11. Display results
print("\n===== MODEL ACCURACY =====")

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("MAPE:", round(mape, 2), "%")


# 12. Display forecast
print("\n===== FORECASTED SALES =====")
print(forecast)


# 13. Plot Actual vs Forecasted Sales
plt.figure(figsize=(12, 6))

plt.plot(
    test.index,
    test.values,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test.index,
    forecast.values,
    marker="o",
    label="Forecasted Sales"
)

plt.title("Actual vs Forecasted Sales - SARIMA")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.tight_layout()


# 14. Save graph
plt.savefig("forecast.png", dpi=300)

plt.show()