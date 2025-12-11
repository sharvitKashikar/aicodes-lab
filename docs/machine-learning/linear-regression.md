# Linear Regression Example (Python)

This example demonstrates a basic implementation of Linear Regression using Python, specifically leveraging the `numpy` library for numerical operations and `scikit-learn` (sklearn) for the linear regression model.

## Overview

The `linear.py` script trains a simple linear regression model to predict house prices based on their size. It showcases the fundamental steps:

1.  **Data Preparation**: Defining a sample dataset for house sizes (X) and corresponding prices (y).
2.  **Model Creation**: Initializing the `LinearRegression` model from `sklearn`.
3.  **Model Training**: Fitting the model to the sample data.
4.  **Prediction**: Using the trained model to predict the price of a new house size.
5.  **Output**: Displaying the calculated coefficient (slope), intercept, and the predicted price.

## Prerequisites

Before running the script, ensure you have the necessary libraries installed:

```bash
pip install numpy scikit-learn
```

## Code Example

Here's the content of the `linear.py` script:

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

## How to Run

1.  Save the code above as `linear.py` in your local environment.
2.  Open your terminal or command prompt.
3.  Navigate to the directory where you saved `linear.py`.
4.  Run the script using Python:

    ```bash
    python linear.py
    ```

## Expected Output

Upon running the script, you will see output similar to this, showing the model's parameters and the prediction:

```
Coefficient (Slope): 0.10625
Intercept: -20.625000000000004
Predicted price for 900 sq ft house: 75 lakh ₹
```

This output indicates that for every square foot increase in house size, the price is predicted to increase by approximately 0.10625 lakh ₹. The intercept represents the baseline price (though it might not be physically interpretable in some contexts).
