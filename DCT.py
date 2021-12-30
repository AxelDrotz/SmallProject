import numpy as np

m = 50
n = 50


def dct(matrix):
    """Finds the discrete cosine transform of an array."""

    transformed_matrix = np.zeros((n, m))

    for i in range(m):
        for j in range(n):
            if i == 0:
                ci = 1/np.sqrt(m)
            else:
                ci = np.sqrt(2/m)
            if j == 0:
                cj = 1/np.sqrt(n)
            else:
                cj = np.sqrt(2/n)

            summation = 0

            for k in range(m):
                for l in range(n):
                    signal = matrix[k][l] \
                             * np.cos((2*k + 1) * i * np.pi / (2*m)) \
                             * np.cos((2*l + 1) * j * np.pi / (2*n))
                    summation += signal

            transformed_matrix[i][j] = ci * cj * summation

    return transformed_matrix

def inv_dct(matrix):
    """Performs the inverse discrete cosine transform on a matrix in the cosine-space."""

    inv_transformed_matrix = np.zeros((n, m))

    for i in range(m):
        for j in range(n):
            if i == 0:
                ci = 1/np.sqrt(m)
            else:
                ci = np.sqrt(2/m)
            if j == 0:
                cj = 1/np.sqrt(n)
            else:
                cj = np.sqrt(2/n)

            summation = 0

            for k in range(m):
                for l in range(n):
                    signal = matrix[k][l] \
                             * np.cos((2*k + 1) * i * np.pi / (2*m)) \
                             * np.cos((2*l + 1) * j * np.pi / (2*n))
                    summation += signal

            inv_transformed_matrix[i][j] = ci * cj * summation

    return inv_transformed_matrix





def main():
    matrix = np.ones((n, m))
    DCT = dct(matrix)
    np.save('transformed_matrix', DCT)

if __name__ == __main__:
    main()