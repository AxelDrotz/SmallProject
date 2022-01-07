import numpy as np
from matplotlib import pyplot as plt
import time
import random


def create_T(n, m):
    """Creates the basis vectors."""
    T = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if i == 0:
                T[i][j] = 1/np.sqrt(n)
            else:
                T[i][j] = np.sqrt(2/n) * np.cos((2*j + 1) * i * np.pi / (2*n))
    return T


def DCT(T, T_inv, col, origin_dim, current_dim):
    """Performs a modified DCT on a vector."""
    data = np.matmul(T, col)
    for removed_dimension in range(origin_dim - current_dim):
        data = np.delete(data, data.argmin())

    data = np.matmul(T_inv, data)

    return data


def dct_error(x_matrix, N, d, k, T, T_inv):
    dct_average_error = 0
    pairs = []

    # Call DCT and observe the computation time
    time_start = time.time()
    time_elapsed = (time.time() - time_start)

    round = 0
    while round < 400:
        sample_flag = False
        while not sample_flag:
            i, j = [random.randint(0, 799) for p in range(0, 2)]
            if ([i, j] not in pairs) and ([j, i] not in pairs) and (i != j):
                pairs.append([i, j])
                sample_flag = True
                round += 1
        # EXTRACT REFERENCE
        xi = x_matrix[:, i]
        xj = x_matrix[:, j]
        x_dist = np.linalg.norm(xi - xj)

        # Calculate DCT average error
        dct_xi = DCT(T, T_inv, xi, d, k)
        dct_xj = DCT(T, T_inv, xj, d, k)
        dct_dist = np.linalg.norm(dct_xi - dct_xj)
        dct_average_error += (dct_dist - x_dist) / x_dist

    return dct_average_error / 400, time_elapsed


def dct_test(matrix, dim, time_list):
    dct_error_res = np.zeros(len(dim))
    d, N = matrix.shape
    T = create_T(d, d)

    for i in range(len(dim)):
        print("k", dim[i])
        k = dim[i]
        T_inv = create_T(k, k)
        dct_error_res[i], elapsed_time = dct_error(matrix, N, d, k, T, T_inv)
        time_list.append(elapsed_time)

    return dct_error_res, time_list


time_list = []
data_matrix = np.load('normalized_array.npy').transpose()
dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 750]
dct_error_results, time_list = dct_test(data_matrix, dims, time_list)

plt.plot(dims, dct_error_results)
plt.xlabel('Reduced dim. of data')
plt.ylabel('Error')
plt.title('Error using DCT')
plt.show()

plt.plot(dims, time_list)
plt.xlabel('Reduced dim. of data')
plt.ylabel('computation time')
plt.title('Computation time for DCT')
plt.show()
