# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create dataset
data = {
    'Age': [25, 35, 45, 23, 52, 40, 28],
    'Income': [30000, 60000, 80000, 25000, 90000, 50000, 32000],
    'CreditScore': [650, 720, 750, 600, 780, 690, 640],
    'Employed': [1, 1, 1, 0, 1, 1, 0],
    'Loan': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert target to numerical
df['Loan'] = df['Loan'].map({'No': 0, 'Yes': 1})

# Features and target
X = df.drop('Loan', axis=1)
y = df['Loan']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Predicted Values:", y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
