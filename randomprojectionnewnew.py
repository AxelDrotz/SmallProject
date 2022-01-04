import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.sparse as sp
import math
import numpy as np
import sys
import scipy.fftpack as scipy
from sklearn import random_projection
X = np.load('normalized_array.npy')
print("dxN")
Xt = np.transpose(X)
print(Xt.shape)

def normalrandomprojection(matrix, k):
    mu, sigma = 0, 1
    R = np.random.normal(mu, sigma, (k, 2500))
    for i in range(len(R)):
        c = np.linalg.norm(R[i])
        R[i] = R[i]/c
    return R
R = normalrandomprojection(Xt, 10)

def test_sparse_random_projection(matrix, dims):
    results_max = np.zeros(len(dims))
    results_avr = np.zeros(len(dims))
    results_min = np.zeros(len(dims))
    results_tim = np.zeros(len(dims))

    d, N = matrix.shape
    #print("N:"+ str(N))
    for i, k in enumerate(dims):

        rp_matrix = normalrandomprojection(matrix, k=k)

        error_avr, error_max, error_min = check_quality(matrix, rp_matrix, N, d, k)
        results_avr[i] = error_avr
        results_max[i] = error_max
        results_min[i] = error_min


    return results_avr#, results_max, results_min


def check_quality(ddim_matrix, kdim_matrix, N, d, k):
    # calculate the error in the distance between members of a pair of data vectors, averaged over 100 pairs
    constant = np.sqrt(d/k)
    average_error = 0
    max_error = 0
    min_error = float('inf')
    pairs = []
    #print("kdim")
    #print(kdim_matrix.shape)
    #print(ddim_matrix.shape, kdim_matrix.shape)
    for _ in range(0, 100):
        # pick random vector pair, but make sure it hasn't been used before
        while True:
            i, j = random.sample(list(range(0, 800)), 2)
            if [i, j] not in pairs:
                pairs.append([i, j])
                break

        xi = ddim_matrix[:, i]
        xj = ddim_matrix[:,j]

        f_xi = np.matmul(kdim_matrix,xi)
        f_xj = np.matmul(kdim_matrix,xj)
        #print(f_xi.shape)
        dist_x = np.linalg.norm(xi-xj)
        dist_fx = constant*np.linalg.norm(f_xi-f_xj)

        error = abs((dist_fx-dist_x)/(dist_x))
        #error = abs((dist_x**2-dist_fx**2)/dist_x**2)
        if max_error < error:
            max_error = error
        if min_error > error:
            min_error = error
        average_error += error

    return average_error/100, max_error, min_error

indx = [i for i in range(1, 10)] + [i*10 for i in range(1,10)] + [i*100 for i in range(1,9)]
#print(indx)

results = test_sparse_random_projection(Xt,indx)
print(results)
print(indx)
plt.plot(indx, results)
plt.show()
