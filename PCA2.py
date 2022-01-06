import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rnd
import time


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
    start = time.time()
    pca_average_error = 0
    pairs = []
    pca_matrix = pca(x_matrix, k).transpose()
    end = time.time()
    print("PCA MATRIX SHAPE")
    print(np.shape(pca_matrix))
    round = 0
    while round < 100:
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
    return pca_average_error/100, end-start


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


# 5  = 800 = observations (images)
# 4 = 2500 = dimensions
image = np.load('normalized_array.npy').transpose()

# org_data = np.array([[1.0, 2, 3, 4], [5, 5, 6, 7], [
#                     1, 4, 2, 3], [5, 3, 2, 1], [8, 1, 2, 2]])
print(np.shape(image))

# k_dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50,
#                 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 750]

k_dimensions = [2, 10, 50, 100, 250, 500, 750]

pca_error_results, pca_time = pca_test(image, k_dimensions)

fig, axis = plt.subplots(2)
axis[0].plot(k_dimensions, pca_error_results)
axis[0].set_title("PCA")
axis[1].set_yscale("log")
axis[1].plot(k_dimensions, pca_time)
axis[1].set_title("PCA Elapsed Time")
plt.show()
