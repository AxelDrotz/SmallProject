import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time

def PCA(A):
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
    return T

def PCA_error(x_matrix, N, d, k):
    averageError = 0
    pairs = []
    rMatrix = normal_rp(2500, k)
    round = 0
    while round < 100:
        sampleFlag = False
        while not sampleFlag:
            i, j = [random.randint(0, 799) for p in range(0, 2)]
            if([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sampleFlag = True
                round += 1
        #EXTRACT REFERENCE
        xi = xMatrix[:, i]
        xj = xMatrix[:,j]
        x_dist = np.linalg.norm(xi-xj)

        #RANDOM PROJECTION
        constant = np.sqrt(d/k)
        rp_xi = np.matmul(rMatrix,xi)
        rp_xj = np.matmul(rMatrix,xj)
        rp_dist = constant*np.linalg.norm(rp_xi-rp_xj)
        rp_average_error += abs((rp_dist-x_dist)/(x_dist))

        #SPARSE RANDOM PROJECTION
        constant = np.sqrt(d/k)
        srp_xi = np.matmul(srp_r_matrix,xi)
        srp_xj = np.matmul(srp_r_matrix,xj)
        srp_dist = constant*np.linalg.norm(srp_xi-srp_xj)
        srp_average_error += abs((srp_dist-x_dist)/(x_dist))
    return rp_average_error/100, srp_average_error/100



dataMatrix = np.load("normalized_array.npy")

timeList = []
kList = list(range(2,802))

# For different dimension reductions k (2500 is the original dimension)
for k in kList:
    time_start = time.time()
    T = PCA(dataMatrix)
    time_elapsed = (time.time() - time_start)
    timeList.append(time_elapsed)
    print(T, T.shape, "where k=", k)

print(timeList)
#del timeList[:1]
print(timeList)

n = 50
timeList2 = [sum(timeList[i:i+n])/n for i in range(0,len(timeList),n)]
kList2 = [sum(kList[i:i+n])/n for i in range(0,len(kList),n)]

np.save("PCA_computation_time", timeList)

print(timeList2, len(timeList2))
plt.plot(kList, timeList)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Computation time for PCA')
plt.show()

plt.scatter(kList2, timeList2)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Computation time for PCA')
plt.show()
