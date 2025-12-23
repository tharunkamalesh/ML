# Import required libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
# Features: Age, Income, LoanAmount, CreditHistory
X = np.array([
    [25, 50000, 200000, 1],
    [35, 60000, 150000, 1],
    [45, 80000, 300000, 1],
    [23, 30000, 250000, 0],
    [40, 40000, 200000, 0],
    [50, 70000, 400000, 0]
])

# Target: Credit Score (Good=1, Bad=0)
y = np.array([1, 1, 1, 0, 0, 0])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
print("Predicted Credit Scores:", y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
