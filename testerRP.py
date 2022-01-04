import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random.mtrand import random_sample
import scipy.sparse as sp
import math
import numpy as np
import sys
import scipy.fftpack as scipy

def normal_rp(matrix, k):
    mu, sigma = 0, 1
    R = np.random.normal(mu, sigma, (k, 2500))
    for i in range(len(R)):
        c = np.linalg.norm(R[i])
        R[i] = R[i]/c
    return R

def sparse_rp(d, k):
    R = np.zeros((k,d))
    for row in range(k):
        for column in range(d):
            p = np.random.binomial(1,2/3)
            if p == 0:
                p = np.random.binomial(1, 0.5)
                if p == 1:
                    R[row][column] = math.sqrt(3)
                else:
                    R[row][column] = - math.sqrt(3)
            else:
                R[row][column] = 0
    for i in range(len(R)):
        c = np.linalg.norm(R[i])
        R[i] = R[i]/c
    return R

def rp_error(x_matrix, N, d, k):
    rp_average_error = 0
    srp_average_error = 0
    pairs = []
    srp_r_matrix = sparse_rp(2500, k)
    rp_r_matrix = normal_rp(2500, k)
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

        #RANDOM PROJECTION
        constant = np.sqrt(d/k)
        rp_xi = np.matmul(rp_r_matrix,xi)
        rp_xj = np.matmul(rp_r_matrix,xj)
        rp_dist = constant*np.linalg.norm(rp_xi-rp_xj)
        rp_average_error += abs((rp_dist-x_dist)/(x_dist))

        #SPARSE RANDOM PROJECTION
        constant = np.sqrt(d/k)
        srp_xi = np.matmul(srp_r_matrix,xi)
        srp_xj = np.matmul(srp_r_matrix,xj)
        srp_dist = constant*np.linalg.norm(srp_xi-srp_xj)
        srp_average_error += abs((srp_dist-x_dist)/(x_dist))
    return rp_average_error/100, srp_average_error/100

def srp_test(matrix, dim):
    rp_error_res = np.zeros(len(dim))
    srp_error_res = np.zeros(len(dim))
    d, N = matrix.shape
    for i in range(len(dim)):
        print(dim[i])
        k=dim[i]
        rp_error_res[i], srp_error_res[i] = rp_error(matrix, N, d, k)
    return rp_error_res, srp_error_res

X = np.load('normalized_array.npy').transpose()
k_dimensions = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,750]
rp_error_results, srp_error_results = srp_test(X,k_dimensions)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(k_dimensions, rp_error_results)
ax1.set_title("Random Projection")
ax2.plot(k_dimensions, srp_error_results)
ax2.set_title("Sparse Random Projection")
plt.show()
