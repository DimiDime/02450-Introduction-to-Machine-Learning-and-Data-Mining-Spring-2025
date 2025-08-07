# exercise 10.1.1

import importlib_resources
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler
from dtuimldmtools import clusterplot

#filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth4.mat")
filename = importlib_resources.files("dtuimldmtools").joinpath("data/faithful.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"].squeeze()]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)

# Number of clusters:
K = 2

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering:
centroids, cls, inertia = k_means(X_scaled, K)

# Plot results:
plt.figure(figsize=(14, 9))
clusterplot(X_scaled, cls, centroids, y)
plt.show()

print("Ran Exercise 10.1.1")

#help(clusterplot)
