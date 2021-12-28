import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing


"""
dataMatrix is a m x n matrix
U = mxm matrix where the coulmns are eigenvectors
"""
U, s, VT = np.linalg.svd(dataMatrix)

# create m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# The reconstruction of dataMatrix, is it the same?
dataMatrix2 = U.dot(Sigma.dot(VT))
print(dataMatrix2)
# transform
T = U.dot(Sigma)
print(T)
T = A.dot(VT.T)
print(T)
