import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time

dataMatrix = np.load("normalized_array.npy")

A = dataMatrix
#A = np.array([[1,2,3], [4,5,6], [7,8,9]]) #Test Matrix

#Standardize
#A = (A - np.mean(A)) / np.std(A)

print("Normalized:", A)
print("DataMatrix has the shape:", A.shape)

"""
dataMatrix is a n x d matrix
U = nxn matrix where the coulmns are eigenvectors
"""

"""
# Select eigenvectors for reduction, n_elements is the new dimension
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# The reconstruction of dataMatrix, is it the same?
dataMatrix2 = U.dot(Sigma.dot(VT))
print(dataMatrix2)
# transform
T = U.dot(Sigma)
# or T = A.dot(VT.T)
print(T, T.shape)
"""

timeList = []
kList = list(range(2,801))

# For different dimension reductions k (2500 is the original dimension)
for k in kList:
    time_start = time.time()
    U, s, VT = np.linalg.svd(A)
    # create m x n Sigma matrix
    Sigma = np.zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
    Sigma = Sigma[:, :k]
    VT = VT[:k, :]
    # The reconstruction of dataMatrix, is it the same?
    dataMatrix2 = U.dot(Sigma.dot(VT))
    # transform
    T = U.dot(Sigma)
    time_elapsed = (time.time() - time_start)
    timeList.append(time_elapsed)
    print(T, T.shape, "where k=", k)

print(timeList)
plt.plot(kList, timeList)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Computation time for PCA')
plt.show()
