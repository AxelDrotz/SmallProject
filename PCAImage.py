import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing

dataMatrix = np.load("image_array.npy")

A = dataMatrix
# Standardizing
A = (A - np.mean(A)) / np.std(A)

print("Standardized:", A)
print("DataMatrix has the shape:", A.shape)

"""
dataMatrix is a n x d matrix
U = nxn matrix where the coulmns are eigenvectors
"""
U, s, VT = np.linalg.svd(A)

# create m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
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
print(T)
