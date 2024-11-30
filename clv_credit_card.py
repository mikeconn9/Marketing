import pandas as pd
import numpy as np

# Simulated Dataset
np.random.seed(42)
data = pd.DataFrame({
    "customer_id": range(1, 1001),
    "age": np.random.randint(18, 70, 1000),
    "income": np.random.randint(30000, 150000, 1000),
    "monthly_transactions": np.random.poisson(20, 1000),
    "avg_transaction_value": np.random.uniform(20, 200, 1000),
    "account_tenure_years": np.random.uniform(0.5, 10, 1000),
    "credit_score": np.random.randint(600, 850, 1000),
    "reward_program": np.random.choice([0, 1], 1000),
    "historical_revenue": np.random.uniform(100, 5000, 1000),
})

# True CLV (Simulated as ground truth for supervised Learning)
data["clv"] = (
    data["historical_revenue"] +
    (data["monthly_transactions"] * data["avg_transaction_value"] * 0.1) +
    (data["income"] * 0.0002)
)

from sklearn.model_selection import train_test_split

# Features and Target
X = data[[
    "age", "income", "monthly_transactions", "avg_transaction_value",
    "account_tenure_years", "credit_score", "reward_program", "historical_revenue"
]]
y = data["clv"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Performance
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"r2: {r2}")