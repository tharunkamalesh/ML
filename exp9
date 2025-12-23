import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 5, 10, 17, 26])

# ---------------- Linear Regression ----------------
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Evaluation
mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

# ---------------- Polynomial Regression (degree=2) ----------------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# Evaluation
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

# ---------------- Results ----------------
print("Linear Regression:")
print("MSE:", mse_linear)
print("R2 Score:", r2_linear)

print("\nPolynomial Regression (Degree 2):")
print("MSE:", mse_poly)
print("R2 Score:", r2_poly)
