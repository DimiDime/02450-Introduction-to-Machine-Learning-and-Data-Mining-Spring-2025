import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances

x=np.array([
  [0,0],
  [1,1],
  [2,1],
  [3,2]
])

d=distance_matrix(x,x,p=2) # 1 := manhattan distance 
print(d)

dd = pairwise_distances(x,x,metric='euclidean')
print(dd)

