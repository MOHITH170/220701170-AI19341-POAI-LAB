# Step 1: Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Step 2: Create a synthetic dataset for clustering
# Creating a dataset with 3 centers (clusters)
X, y = make_blobs(n_samples=500, centers=3, random_state=42)

# Step 3: Preprocess the data (Standardize it for better clustering results)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method to find the optimal number of clusters
wcss = []  # List to store the within-cluster sum of squares (WSS)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.grid(True)
plt.show()

# From the Elbow Method plot, choose the optimal k (let's assume it's 3)
optimal_k = 3

# Step 5: Implement K-Means Clustering with the optimal k (k=3)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Step 6: Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 7: Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label="Centroids")
plt.title("K-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()









Explanation:
Dataset Creation:

We use make_blobs() to create a synthetic dataset with 500 samples and 3 centers (clusters). The dataset has 2 features, making it easy to visualize in a 2D plot.
Preprocessing:

The StandardScaler() is used to standardize the data before applying K-Means, as K-Means is sensitive to the scale of the data.
Elbow Method:

The Elbow Method helps us determine the optimal number of clusters. We train the K-Means model with k ranging from 1 to 10 and plot the Within-Cluster Sum of Squares (WSS) for each k.
The "elbow" in the plot indicates the optimal number of clusters, where adding more clusters doesn't significantly improve the clustering.
K-Means Clustering:

We apply K-Means with the optimal number of clusters (in this case, k=3, determined by the Elbow Method).
The kmeans.cluster_centers_ gives the coordinates of the cluster centroids.
The kmeans.labels_ assigns each data point to a cluster.
Visualization:

A scatter plot is created to visualize the clustered data points. The colors represent different clusters, and the red "x" markers represent the cluster centroids.
Example Output:
Elbow Method Plot: The plot will show the within-cluster sum of squares for different values of k. The optimal k can be found by identifying the "elbow" point in the plot.

K-Means Clustering Result: After determining k=3 from the elbow method, the program will create a scatter plot of the data points colored according to their clusters. The cluster centroids will be marked with red "x" markers.

Notes:
In this example, the synthetic dataset was created with 3 centers, so the Elbow Method will likely indicate k=3 as the optimal number of clusters.
You can replace make_blobs() with your real dataset by loading it using pandas or other data sources.
The Elbow Method graph will help you determine the best k if you are unsure of how many clusters to choose.
