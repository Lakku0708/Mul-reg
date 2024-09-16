import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load dataset (adjust file path as needed)
df = pd.read_csv('50_Startups.csv')

# Data Preprocessing
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']

# Encode 'State' (categorical feature)
X = pd.get_dummies(X, drop_first=True)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single Linear Regression (using 'R&D Spend')
X_single = X_train[['R&D Spend']]
X_single_test = X_test[['R&D Spend']]
regressor_single = LinearRegression()
regressor_single.fit(X_single, y_train)

# Predict and evaluate single regression
y_pred_single = regressor_single.predict(X_single_test)
r2_single = r2_score(y_test, y_pred_single)
print(f"Single Regression R²: {r2_single:.4f}")

# Multiple Linear Regression (using all features)
regressor_multi = LinearRegression()
regressor_multi.fit(X_train, y_train)

# Predict and evaluate multiple regression
y_pred_multi = regressor_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
print(f"Multiple Regression R²: {r2_multi:.4f}")

# Sum of Squared Errors (SSE)
sse_single = np.sum((y_test - y_pred_single)**2)
sse_multi = np.sum((y_test - y_pred_multi)**2)
print(f"Single Regression SSE: {sse_single:.4f}")
print(f"Multiple Regression SSE: {sse_multi:.4f}")
