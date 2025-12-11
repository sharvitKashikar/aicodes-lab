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
