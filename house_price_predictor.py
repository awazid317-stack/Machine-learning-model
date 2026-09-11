# PROJECT 1 HOUSE PRICE PREDICTOR

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

hsng = fetch_california_housing()

df = pd.DataFrame(hsng.data, columns=hsng.feature_names)
df["Price"] = hsng.target

# DATA VISUALIZATION

# df.hist(
#     figsize=(15,10),
#     bins=30,
#     edgecolor="Black"
# )

# plt.tight_layout()
# plt.show()


# TRAIN_TEST

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAINING OF MODEL (Linear Regression)

model = LinearRegression()
model.fit(X_train, y_train)


# PREDICTIONS (Linear Regression)

y_pred_lr = model.predict(X_test)

# EVALUATIONS (Linear Regression)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_score_lr = r2_score(y_test, y_pred_lr)  # Renamed variable to avoid conflict

print("Linear Regression MAE:", mae_lr)
print("Linear Regression R2 Score:", r2_score_lr)


# TRAINING OF MODEL (Random Forest Regressor)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# PREDICTIONS (Random Forest Regressor)
y_pred_rf = rf.predict(X_test)

# EVALUATIONS (Random Forest Regressor)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_score_rf = r2_score(y_test, y_pred_rf)  # Renamed variable and used rf predictions

print("Random Forest MAE:", mae_rf)
print("Random Forest R2 Score:", r2_score_rf)

# VISUALIZATION AND COMPARISSION
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual price")
plt.ylabel("predicted price")

plt.title("Actual vs predicted house price")

plt.show()
