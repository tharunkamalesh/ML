# Import required libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset (Experience vs Salary)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 40000, 45000, 50000])

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

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted Salary:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
