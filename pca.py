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
