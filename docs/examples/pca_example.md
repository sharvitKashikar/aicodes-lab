# Principal Component Analysis (PCA) Example

This example demonstrates how to perform Principal Component Analysis (PCA) for dimensionality reduction using Python's `scikit-learn` library, visualize the results with `matplotlib`, and handle data with `numpy`.

PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This example applies PCA to the well-known Iris dataset to reduce its 4 features down to 2 principal components for easy visualization.

## Code (`pca.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# Apply PCA (4 features → 2 components)
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print("Original Shape:", X.shape)
print("Transformed Shape:", X_transformed.shape)
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# Scatter plot
plt.figure(figsize=(8, 6))
for class_value in np.unique(y):
    plt.scatter(
        X_transformed[y == class_value, 0],
        X_transformed[y == class_value, 1],
        label=labels[class_value]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset (4D → 2D)")
plt.legend()
plt.grid(True)
plt.show()
```

## Explanation

1.  **Libraries Used**: The script imports `numpy` for numerical operations, `matplotlib.pyplot` for plotting, and `load_iris`, `PCA` from `sklearn.datasets` and `sklearn.decomposition` respectively for dataset loading and PCA functionality.

2.  **Dataset Loading**: The `load_iris()` function is used to fetch the Iris dataset, a classic dataset in machine learning and pattern recognition. `X` stores the feature data, `y` stores the target labels, and `labels` holds the names of the target classes.

3.  **Applying PCA**: An instance of `PCA` is created with `n_components=2`, meaning the original 4-dimensional data will be reduced to 2 dimensions. The `fit_transform()` method fits the PCA model to the data and then transforms `X` into its 2-dimensional principal components `X_transformed`.

4.  **Output**: The script prints the original and transformed shapes of the data, demonstrating the dimensionality reduction. It also prints the `explained_variance_ratio_`, which indicates the proportion of total variance explained by each principal component.

5.  **Visualization**: A scatter plot is generated to visualize the transformed data. Each class in the Iris dataset is plotted with a distinct color, allowing clear visual separation of the clusters in the 2D principal component space.

## How to Run

To run this example, save the code as `pca.py` and execute it from your terminal:

```bash
python pca.py
```

You will need to have `numpy`, `matplotlib`, and `scikit-learn` installed. You can install them via pip:

```bash
pip install numpy matplotlib scikit-learn
```
