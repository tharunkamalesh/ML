# Import required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Year': [2014, 2013, 2017, 2011, 2018],
    'PresentPrice': [5.59, 9.54, 7.27, 3.10, 8.18],
    'KmsDriven': [27000, 43000, 6900, 52000, 12000],
    'FuelType': [0, 1, 0, 0, 1],          # Petrol=0, Diesel=1
    'SellerType': [1, 1, 1, 0, 1],        # Dealer=1, Individual=0
    'Transmission': [0, 0, 0, 0, 1],      # Manual=0, Automatic=1
    'SellingPrice': [3.35, 5.50, 4.75, 2.85, 6.80]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('SellingPrice', axis=1)
y = df['SellingPrice']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted Price:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
