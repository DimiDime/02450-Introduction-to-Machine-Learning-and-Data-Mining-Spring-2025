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

import scipy.stats as st
import sklearn.linear_model
import sklearn.tree






# requires data from exercise 1.5.1
#from ex7_2_1 import *
from sklearn import model_selection

from dtuimldmtools import *
from dtuimldmtools.statistics.statistics import correlated_ttest

loss = 2
X,y = X[:,:10], X[:,10:]
# This script crates predictions from three KNN classifiers using cross-validation

K = 10 # We presently set J=K
m = 1
r = []
kf = model_selection.KFold(n_splits=K)

for dm in range(m):
    y_true = []
    yhat = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        mA = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
        mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

        yhatA = mA.predict(X_test)
        yhatB = mB.predict(X_test)[:, np.newaxis]  # justsklearnthings
        y_true.append(y_test)
        yhat.append( np.concatenate([yhatA, yhatB], axis=1) )

        r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)

if m == 1:
    y_true = np.concatenate(y_true)[:,0]
    yhat = np.concatenate(yhat)

    # note our usual setup I ttest only makes sense if m=1.
    zA = np.abs(y_true - yhat[:,0] ) ** loss
    zB = np.abs(y_true - yhat[:,1] ) ** loss
    z = zA - zB

    CI_setupI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_setupI = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

    print("p-setup II: ", p_setupII)
    print("p-setup I: ", p_setupI)
    print("CI-setup II: ", CI_setupII)
    print("CI-setup I: ", CI_setupI)