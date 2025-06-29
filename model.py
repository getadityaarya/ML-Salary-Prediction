import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv('Salary Data.csv')

# Cleaning and preprocessing
data = data.dropna()

# Encode categorical features (e.g., industry)
if data['industry'].dtype == object:
    data = pd.get_dummies(data, columns=['industry'], drop_first=True)

# Features and target
X = data.drop('salary', axis=1)
y = data['salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')
