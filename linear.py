# Linear Regression Example
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
# X = house size (sq ft), y = price (in lakh ₹)
X = np.array([[500], [700], [800], [1000], [1200]])
y = np.array([30, 45, 50, 65, 80])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict price of a 900 sq ft house
prediction = model.predict([[900]])

print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted price for 900 sq ft house:", prediction[0], "lakh ₹")
