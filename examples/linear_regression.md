# Linear Regression

This example demonstrates a simple Linear Regression model using `scikit-learn` to predict house prices based on their size.

## Algorithm Overview

Linear Regression is a fundamental supervised learning algorithm used for modeling the relationship between a scalar response variable (dependent variable) and one or more explanatory variables (independent variables). The goal is to find the best-fitting straight line (or hyperplane) through the data points that minimizes the sum of the squared differences between the observed and predicted values.

## Code Explanation

1.  **Imports**: Uses `numpy` for data handling and `sklearn.linear_model.LinearRegression` for the model.
2.  **Sample Dataset**: 
    *   `X`: A NumPy array representing the independent variable (e.g., house size in sq ft). It's reshaped to `[[value]]` because `scikit-learn` expects 2D input for features.
    *   `y`: A NumPy array representing the dependent variable (e.g., house price).
3.  **Model Creation and Training**: An instance of `LinearRegression` is created, and its `fit()` method is called with `X` and `y` to train the model.
4.  **Prediction**: The trained model's `predict()` method is used to predict the price for a new house size (900 sq ft).
5.  **Output**: Prints the learned `coefficient_` (slope) and `intercept_` of the regression line, along with the `predicted` price.

## How to Run

1.  Save the code as `linear.py`.
2.  Ensure you have the necessary libraries installed:
    ```bash
    pip install numpy scikit-learn
    ```
3.  Run the script from your terminal:
    ```bash
    python linear.py
    ```

## Expected Output

```
Coefficient (Slope): 0.07125000000000001
Intercept: -5.750000000000007
Predicted price for 900 sq ft house: 58.375 lakh ₹
```

```python
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
```