from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from numpy.random.mtrand import random_sample
import scipy.sparse as sp
import math
import numpy as np
import numpy.linalg as la
import scipy.fftpack as scipy
import time


def normal_rp(d, k):
    mu, sigma = 0, 1
    R = np.zeros((k, d))
    for row in range(k):
        for column in range(d):
            R[row][column] = np.random.normal(mu, sigma)
    for i in range(len(R)):
        c = np.linalg.norm(R[i])
        R[i] = R[i]/c
    return R


def sparse_rp(d, k):
    R = np.zeros((k, d))
    #R = math.sqrt(3)*(np.random.binomial(1, 1/3, (k, d)))
    for row in range(k):
        for column in range(d):
            p = np.random.binomial(1, 2/3)
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


def srp_error(x_matrix, N, d, k):
    run_time = 0
    start = time.time()
    srp_average_error = 0
    pairs = []
    srp_r_matrix = sparse_rp(d, k)
    round = 0
    end = time.time()
    r_time = end - start
    while round < 100:
        start = time.time()
        sample_flag = False
        while not sample_flag:
            i, j = [rnd.randint(0, N-1) for p in range(0, 2)]
            if([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sample_flag = True
                round += 1
        # EXTRACT REFERENCE
        xi = x_matrix[:, i]
        xj = x_matrix[:, j]
        x_dist = np.linalg.norm(xi-xj)
        # SPARSE RANDOM PROJECTION
        constant = np.sqrt(d/k)
        srp_xi = np.matmul(srp_r_matrix, xi)
        srp_xj = np.matmul(srp_r_matrix, xj)
        srp_dist = constant*np.linalg.norm(srp_xi-srp_xj)
        srp_average_error += ((srp_dist-x_dist)/(x_dist))
        end = time.time()
        run_time += end - start + r_time

    return (srp_average_error/100), run_time/100


def rp_error(x_matrix, N, d, k):
    run_time = 0
    start = time.time()
    rp_average_error = 0
    pairs = []
    rp_r_matrix = normal_rp(d, k)
    round = 0
    end = time.time()
    r_time = end - start
    while round < 100:
        start = time.time()
        sample_flag = False
        while not sample_flag:
            i, j = [rnd.randint(0, N-1) for p in range(0, 2)]
            if([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sample_flag = True
                round += 1
        # EXTRACT REFERENCE
        xi = x_matrix[:, i]
        xj = x_matrix[:, j]
        x_dist = np.linalg.norm(xi-xj)

        # RANDOM PROJECTION
        constant = np.sqrt(d/k)
        rp_xi = np.matmul(rp_r_matrix, xi)
        rp_xj = np.matmul(rp_r_matrix, xj)
        rp_dist = constant*np.linalg.norm(rp_xi-rp_xj)
        rp_average_error += ((rp_dist-x_dist)/(x_dist))

        # SPARSE RANDOM PROJECTION
        end = time.time()
        run_time += end - start + r_time
    return (rp_average_error/100), run_time/100


def rp_srp_test(matrix, dim):
    rp_error_res = np.zeros(len(dim))
    srp_error_res = np.zeros(len(dim))
    rp_time = np.zeros(len(dim))
    srp_time = np.zeros(len(dim))
    d, N = matrix.shape
    for i in range(len(dim)):
        print(dim[i])
        k = dim[i]
        srp_error_res[i], srp_time[i] = srp_error(
            matrix, N, d, k)
        rp_error_res[i], rp_time[i] = rp_error(
            matrix, N, d, k)
    return rp_error_res, srp_error_res, rp_time, srp_time


def pca(org_data, custom_k):
    # Standardize the Dataset
    org_data = org_data.transpose()
    k, d = np.shape(org_data)
    for i in range(d):
        mean = np.mean(org_data[:, i])
        std = np.std(org_data[:, i], ddof=1)
        org_data[:, i] = ((org_data[:, i] - mean)/float(std))
    # Compute covariance matrix
    cov = np.cov(org_data.transpose(), ddof=0)
    eig_val, eig_vec = la.eig(cov)

    # Select Dimension Values
    k_eig_val = eig_val[:int(custom_k)]
    k_eig_vec = eig_vec[:, :int(custom_k)]

    res = np.matmul(org_data, k_eig_vec)
    return res


def pca_error(x_matrix, N, d, k):
    run_time = 0
    start = time.time()
    pca_average_error = 0
    pairs = []
    pca_matrix = pca(x_matrix, k).transpose()
    round = 0
    end = time.time()
    c_time = end - start
    while round < 100:
        start = time.time()
        sample_flag = False
        while not sample_flag:
            i, j = [rnd.randint(0, N-1) for p in range(0, 2)]
            if([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sample_flag = True
                round += 1
        # EXTRACT REFERENCE
        xi = x_matrix[:, i]
        xj = x_matrix[:, j]
        x_dist = np.linalg.norm(xi-xj)
        # SPARSE RANDOM PROJECTION
        pca_xi = pca_matrix[:, i]
        pca_xj = pca_matrix[:, j]
        pca_x_dist = np.linalg.norm(pca_xi-pca_xj)
        pca_average_error += ((pca_x_dist-x_dist)/(x_dist))
        end = time.time()
        run_time += end - start + c_time
    return (pca_average_error/100), run_time/100


def pca_test(matrix, dim):
    pca_error_res = np.zeros(len(dim))
    pca_time = np.zeros(len(dim))
    d, N = matrix.shape
    for i in range(len(dim)):
        print(dim[i])
        k = dim[i]
        pca_error_res[i], pca_time[i] = pca_error(
            matrix, N, d, k)
    return pca_error_res, pca_time

# MAIN


image = np.load('normalized_array.npy').transpose()
imagePaper = np.load('paper_array.npy')
text = np.load('./ConvertTxtData/textdata.npy')
print("Image", np.shape(image))
print("Image Paper", np.shape(imagePaper))
print("Text", np.shape(text))

k_dimensions = [2, 10, 50, 100, 250, 500, 750]

print("RANDOM PROJECTIONS")
rp_error_results, srp_error_results, rp_time, srp_time = rp_srp_test(
    image, k_dimensions)
print("PCA")
pca_error_results, pca_time = pca_test(image, k_dimensions)

print("RP TIME", np.sum(rp_time))
print("SRP TIME", np.sum(srp_time))
print("PCA TIME", np.sum(pca_time))
fig, axis = plt.subplots(2, 4)
axis[0, 0].plot(k_dimensions, rp_error_results)
axis[0, 0].set_title("RP")
axis[0, 1].plot(k_dimensions, srp_error_results)
axis[0, 1].set_title("SRP")
axis[1, 0].set_yscale("log")
axis[1, 0].plot(k_dimensions, rp_time)
axis[1, 0].set_title("RP Elapsed Time")
axis[1, 1].set_yscale("log")
axis[1, 1].plot(k_dimensions, srp_time)
axis[1, 1].set_title("SRP Elapsed Time")
axis[0, 2].plot(k_dimensions, pca_error_results)
axis[0, 2].set_title("PCA")
axis[1, 2].set_yscale("log")
axis[1, 2].plot(k_dimensions, pca_time)
axis[1, 2].set_title("PCA Elapsed Time")
axis[0, 3].plot(k_dimensions, rp_error_results)
axis[0, 3].plot(k_dimensions, srp_error_results)
axis[0, 3].plot(k_dimensions, pca_error_results)
axis[0, 3].set_title("ALL")
axis[0, 3].legend(["RP", "SRP", "PCA"])
axis[1, 3].set_yscale("log")
axis[1, 3].plot(k_dimensions, rp_time)
axis[1, 3].plot(k_dimensions, srp_time)
axis[1, 3].plot(k_dimensions, pca_time)
axis[1, 3].set_title("ALL Elapsed Time")
axis[1, 3].legend(["RP", "SRP", "PCA"])

plt.show()
