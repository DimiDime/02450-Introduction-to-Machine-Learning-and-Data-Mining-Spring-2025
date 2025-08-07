# exercise 5.1.5
import os
import importlib_resources
import numpy as np
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
filename = importlib_resources.files("dtuimldmtools").joinpath("data/wine.mat")
workingDir = os.getcwd()
print("Running from: " + workingDir)

# Pick the relevant variables
mat_data = loadmat(filename)
X = mat_data["X"]
y = mat_data["y"].astype(int).squeeze()
C = mat_data["C"][0, 0]
M = mat_data["M"][0, 0]
N = mat_data["N"][0, 0]

attributeNames = [i[0][0] for i in mat_data["attributeNames"]]
classNames = [j[0] for i in mat_data["classNames"] for j in i]

# Remove outliers
outlier_mask = (X[:, 1] > 20) | (X[:, 7] > 10) | (X[:, 10] > 200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask, :]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:, 0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

print("Ran Exercise 5.1.5 - loading the Wine data")

import numpy as np
import scipy.stats
import scipy.stats as st
import sklearn.tree

# requires data from exercise 1.5.1
#from ex5_1_5 import *
from sklearn import model_selection

X, y = X[:, :10], X[:, 10:]
# This script crates predictions from three KNN classifiers using cross-validation

test_proportion = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_proportion
)

mA = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

yhatA = mA.predict(X_test)
yhatB = mB.predict(X_test)[:, np.newaxis]  #  justsklearnthings

# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - yhatA) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(
    1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA)
)  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_test - yhatB) ** 2
z = zA - zB
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

# Note: You may want to print the values here!
print("p-val: ", p)
print('z-val: ', z)
print("CI: ", CI)