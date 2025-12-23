# Import required libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
# Features: Area, Bedrooms, Bathrooms
X = np.array([
    [800, 1, 1],
    [1000, 2, 1],
    [1200, 2, 2],
    [1500, 3, 2],
    [1800, 3, 3]
])

# Target: House Price (in Lakhs)
y = np.array([40, 50, 65, 80, 95])

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted House Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
