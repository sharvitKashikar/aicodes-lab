# K-Means Clustering

This example demonstrates the K-Means clustering algorithm, an unsupervised machine learning algorithm used for partitioning `n` observations into `k` clusters where each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

## Algorithm Overview

K-Means aims to partition `n` data points into `k` clusters such that each data point belongs to the cluster with the closest mean. It works iteratively to assign each data point to one of `k` clusters based on the feature similarity. The clusters are then refined by computing the mean of all data points within each cluster and setting these means as the new cluster centroids.

## Code Explanation

1.  **Imports**: Uses `numpy` for numerical operations, `sklearn.cluster.KMeans` for the clustering algorithm, and `matplotlib.pyplot` for visualization.
2.  **Sample Dataset (`X`)**: A 2D NumPy array representing sample data points, where each point has two features (e.g., Age and Income).
3.  **K-Means Initialization**: An instance of `KMeans` is created with `n_clusters=3` to find three distinct clusters and `random_state=0` for reproducibility.
4.  **Training**: The `fit()` method is called on the sample data `X` to perform the clustering.
5.  **Output**: Prints the coordinates of the `cluster_centers_` (means of each cluster) and the `labels_` array, which indicates the cluster assignment for each data point.
6.  **Visualization**: Uses `matplotlib` to plot the original data points, colored according to their assigned cluster. The cluster centers are marked with an 'x'.

## How to Run

1.  Save the code as `K_means.py`.
2.  Ensure you have the necessary libraries installed:
    ```bash
    pip install numpy scikit-learn matplotlib
    ```
3.  Run the script from your terminal:
    ```bash
    python K_means.py
    ```

## Expected Output

The script will print the cluster centers and labels, and then display a scatter plot visualizing the clustered data.

```
Cluster Centers:
 [[27.         22000.     ]
 [47.         52000.     ]
 [67.         72333.33333333]]
Labels: [0 0 0 1 1 1 2 2 2]
```
(A matplotlib plot window will also appear.)

```python
# K-Means Clustering Example
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset (Age, Income)
X = np.array([
    [25, 20000], [27, 22000], [29, 24000],
    [45, 50000], [47, 52000], [49, 54000],
    [65, 70000], [67, 72000], [69, 75000]
])

# Apply KMeans for 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)

# Plot
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', s=200)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")
plt.show()
```