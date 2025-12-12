# K-Means Clustering Example

This document describes the `K means.py` script, which demonstrates a basic implementation of the K-Means clustering algorithm using `scikit-learn`.

## Purpose

K-Means is a popular unsupervised machine learning algorithm used for partitioning `n` observations into `k` clusters, where each observation belongs to the cluster with the nearest mean (cluster center), serving as a prototype of the cluster. This example illustrates how to apply K-Means to a simple dataset and visualize the results.

## Script Overview

The `K means.py` script performs the following steps:

1.  **Imports necessary libraries**: `numpy` for numerical operations, `sklearn.cluster.KMeans` for the clustering algorithm, and `matplotlib.pyplot` for plotting.
2.  **Defines a sample dataset `X`**: This dataset consists of 9 data points, each with two features: 'Age' and 'Income'. These points are designed to naturally form three distinct groups.
3.  **Applies K-Means Clustering**: It initializes and fits a `KMeans` model with `n_clusters=3` (indicating the desire to find 3 clusters) and a `random_state` for reproducibility.
4.  **Prints Results**: The script outputs the coordinates of the identified cluster centers and the cluster label assigned to each data point.
5.  **Visualizes Clusters**: It generates a scatter plot to visually represent the data points, colored according to their assigned clusters. The cluster centers are also marked on the plot.

## How to Run the Example

### Prerequisites

Ensure you have the following Python libraries installed:

*   `numpy`
*   `scikit-learn`
*   `matplotlib`

You can install them using pip:

```bash
pip install numpy scikit-learn matplotlib
```

### Execution

1.  Save the provided code as `K means.py` (or ensure it's in your working directory).
2.  Run the script from your terminal:

    ```bash
    python "K means.py"
    ```

## Interpreting the Output

Upon execution, the script will print the following to the console:

```
Cluster Centers:
 [[27. 22000.]
 [47. 52000.]
 [67. 72333.33333333]]
Labels: [0 0 0 1 1 1 2 2 2]
```

*   **`Cluster Centers`**: These are the coordinates (average age and income) for the center of each of the 3 identified clusters. For instance, the first cluster center is at approximately (27 Age, 22000 Income).
*   **`Labels`**: This array indicates which cluster each original data point belongs to. For example, the first three data points ([25, 20000], [27, 22000], [29, 24000]) are all assigned to cluster 0.

Additionally, a plot titled "K-Means Clustering" will be displayed, showing the data points colored by their cluster and the cluster centers marked with 'x'.

## Code Example (`K means.py`)

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