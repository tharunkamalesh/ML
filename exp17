# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'RAM': [4, 6, 8, 3, 12],
    'Storage': [64, 128, 256, 32, 512],
    'Battery': [3000, 3500, 4000, 2500, 4500],
    'Screen': [6.1, 6.5, 6.7, 5.8, 6.9],
    'Camera': [12, 48, 64, 8, 108],
    'Price': [300, 450, 700, 200, 1000]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
