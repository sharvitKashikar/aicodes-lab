# Support Vector Machine (SVM) Classification Example

This example demonstrates how to implement a basic Support Vector Machine (SVM) classifier for a classification task using the `scikit-learn` library in Python.

## Overview

The script `svm.py` loads the famous Iris dataset, splits it into training and testing sets, trains a linear kernel SVM model, makes predictions, and evaluates the model's accuracy and performance using a classification report.

## Prerequisites

To run this example, you need Python and `scikit-learn` installed. You can install `scikit-learn` using pip:

```bash
pip install scikit-learn
```

## How to Run

1. Navigate to the root directory of the repository.
2. Execute the `svm.py` script:

```bash
python svm.py
```

## Code Example (`svm.py`)

```python
# Support Vector Machine (SVM) Classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data      # features
y = iris.target    # labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create SVM classifier
svm = SVC(kernel='linear')   # linear kernel SVM
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Expected Output

When you run the script, you will see output similar to this (exact numbers might vary slightly based on scikit-learn version but the structure will be similar):

```text
Accuracy: 1.0

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

```

## Further Exploration

- Experiment with different SVM kernels (e.g., `rbf`, `poly`) by changing the `kernel` parameter in `SVC`.
- Adjust the `test_size` or `random_state` in `train_test_split` to see how it affects the model's performance.
- Explore other datasets available in `sklearn.datasets`.
