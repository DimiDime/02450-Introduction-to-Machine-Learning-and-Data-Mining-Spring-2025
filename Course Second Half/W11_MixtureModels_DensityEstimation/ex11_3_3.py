# Exercise 11.3.1 (KNN Version): Outlier detection using KNN density and average relative density
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Generate 1D Gaussian mixture data + one outlier
N = 1000
M = 1
X = np.empty((N, M))
means = np.array([1, 3, 6])
stds = np.array([1, 0.5, 2])
component_sizes = np.random.multinomial(N, [1/3]*3)

for i, size in enumerate(component_sizes):
    X[component_sizes.cumsum()[i] - size : component_sizes.cumsum()[i]] = \
        np.random.normal(means[i], np.sqrt(stds[i]), (size, M))

# Add one extreme outlier
X[-1, 0] = -10

# Number of neighbors
K = 50

# Fit KNN and compute densities
knn = NearestNeighbors(n_neighbors=K).fit(X)
distances, _ = knn.kneighbors(X)

# KNN density: inverse of average distance to neighbors (excluding self at index 0)
knn_density = 1.0 / (distances[:, 1:].sum(axis=1) / (K - 1))

# KNN average relative density
ref_densities = knn_density[_[:, 1:]]  # get neighbors' densities
knn_avg_rel_density = knn_density / (ref_densities.sum(axis=1) / (K - 1))

# Sort and find potential outliers
density_sorted_idx = knn_density.argsort()
rel_density_sorted_idx = knn_avg_rel_density.argsort()

# -------------------- PLOTS --------------------

plt.figure(figsize=(10, 6))
plt.bar(range(20), knn_density[density_sorted_idx[:20]])
plt.title("Outlier score (KNN density)")
plt.xlabel("Sample index (lowest density)")
plt.ylabel("Density estimate")

plt.figure(figsize=(10, 6))
plt.bar(range(20), knn_avg_rel_density[rel_density_sorted_idx[:20]])
plt.title("Outlier score (KNN average relative density)")
plt.xlabel("Sample index (lowest average relative density)")
plt.ylabel("Relative density score")

plt.show()

print(f"Lowest KNN density point index: {density_sorted_idx[0]}")
print(f"Lowest relative density point index: {rel_density_sorted_idx[0]}")
