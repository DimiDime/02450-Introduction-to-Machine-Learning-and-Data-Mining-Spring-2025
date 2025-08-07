# exercise 10.2.1
import importlib_resources
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from dtuimldmtools import clusterplot

#filename = importlib_resources.files("dtuimldmtools").joinpath("data/synth1.mat")
filename = importlib_resources.files("dtuimldmtools").joinpath("data/faithful.mat")

# Load Matlab data file and extract variables of interest
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].squeeze()
attributeNames = [name[0] for name in mat_data["attributeNames"].squeeze()]
classNames = [name[0][0] for name in mat_data["classNames"]]
N, M = X.shape
C = len(classNames)


# Perform hierarchical/agglomerative clustering on data matrix
Method = "ward" #"single"complete
Metric = "euclidean"#"euclidean"cosine"minkowski

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4
cls = fcluster(Z, criterion="maxclust", t=Maxclust)
plt.figure(1)
clusterplot(X_scaled, cls.reshape(cls.shape[0], 1), y=y)

# Display dendrogram
max_display_levels = 6
plt.figure(2, figsize=(10, 4))
dendrogram(
    Z, truncate_mode="level", p=max_display_levels, color_threshold=Z[-Maxclust + 1, 2]
)

plt.show()

print("Ran Exercise 10.2.1")
