import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load dataset
data = pd.read_csv("sales_data.csv")

# 2. Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# 3. Calculate monthly sales
monthly_sales = data.groupby(
    data["Date"].dt.to_period("M")
)["Sales"].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()

# 4. Perform seasonal decomposition
result = seasonal_decompose(
    monthly_sales,
    model="additive",
    period=12
)

# 5. Plot decomposition
result.plot()

plt.tight_layout()

# 6. Save graph
plt.savefig("seasonality.png", dpi=300)

# 7. Display graph
plt.show()