import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time

def execPCA(A, k):
    #Step1
    A_meaned = A - np.mean(A , axis = 0)

    #Step2
    cov_mat = np.cov(A_meaned , rowvar = False)

    #Step3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

    #Step4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    #Step5
    eigenvector_subset = sorted_eigenvectors[:,0:k]

    #Step6
    R_matrix = np.dot(eigenvector_subset.transpose() , A_meaned.transpose())#.transpose()
    return R_matrix

def pca_error(x_matrix, N, d, k):
    pca_average_error = 0
    pairs = []

    # Call PCA and observe the computation time
    time_start = time.time()
    r_matrix = execPCA(x_matrix, k)
    time_elapsed = (time.time() - time_start)

    round = 0
    while round < 100:
        sample_flag = False
        while not sample_flag:
            i, j = [random.randint(0, 799) for p in range(0, 2)]
            if([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sample_flag = True
                round += 1
        #EXTRACT REFERENCE
        xi = x_matrix[:, i]
        xj = x_matrix[:,j]
        x_dist = np.linalg.norm(xi-xj)

        # Calculate PCA average error
        pca_xi = r_matrix[:,i]
        pca_xj = r_matrix[:,j]
        pca_dist = np.linalg.norm(pca_xi-pca_xj)
        pca_average_error += (pca_dist-x_dist)/(x_dist)

    return pca_average_error/100, time_elapsed


def pca_test(matrix, dim, timeList):
    pca_error_res = np.zeros(len(dim))
    d, N = matrix.shape

    for i in range(len(dim)):
        print("k", dim[i])
        k=dim[i]
        pca_error_res[i], elapsed_time = pca_error(matrix, N, d, k)
        timeList.append(elapsed_time)

    return pca_error_res, timeList


dataMatrix = np.load("normalized_array.npy").transpose()
print("datamatrix:", dataMatrix, dataMatrix.shape)

timeList = []
k_dimensions = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,750]
pca_error_results, time = pca_test(dataMatrix, k_dimensions, timeList)
print("pca_error_results:", pca_error_results)

plt.plot(k_dimensions, pca_error_results)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Error using PCA')
plt.show()

#np.save("pca_error_results", pca_error_results)

plt.plot(k_dimensions, timeList)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Computation time for PCA')
plt.show()


"""
timeList = []
kList = list(range(2,802))

# For different dimension reductions k (2500 is the original dimension)
for k in kList:
    time_start = time.time()
    T = PCA(dataMatrix, k)
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
"""
