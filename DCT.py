import numpy as np

m = 50
n = 50

section_size = 5

N = n // section_size


def create_T():
    """Creates the basis vectors."""
    T = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == 0:
                T[i][j] = 1/np.sqrt(N)
            else:
                T[i][j] = np.sqrt(2/N) * np.cos((2*j + 1) * i * np.pi / (2*N))
    return T


def main():
    # Defining the quantization matrix from
    # https://www.researchgate.net/figure/Labelling-of-DCT-coefficients-of-5x5-image-block_fig1_224401255:
    Q = np.matrix([[0, 1, 5, 6, 14], [2, 4, 7, 13, 15], [3, 8, 12, 16, 21], [9, 11, 17, 20, 22], [10, 18, 19, 23, 24]])


    # Importing the data:
    vector_vals = np.load('datafile_name')
    pic_matrix = np.reshape(vector_vals, (n, m))

    # Create the basis vectors:
    T = create_T()

    # Center data around 0:
    M = pic_matrix - 0.5

    # Create sub_matrices:
    blocks = np.zeros((N, N))
    Ds = blocks.copy()

    for i in range(N):
        for j in range(N):
            blocks[i][j] = M[i*section_size:(i+1)*section_size, j*section_size:(j+1)*section_size].copy()
            Ds[i][j] = np.matmul(T, np.matmul(blocks[i][j], T.transpose()))

    # We now have the blocks in cosine-space, and need to project them on the quantization matrix.
    Cs = blocks.copy()
    Rs = blocks.copy()
    Ns = blocks.copy()
    for i in range(N):
        for j in range(N):
            M = Ds[i][j]
            for iprim in range(section_size):
                for jprim in range(section_size):
                    Cs[i][j][iprim][jprim] = round(M[iprim][jprim]/Q[iprim][jprim])
                    Rs[i][j][iprim][jprim] = Q[iprim][jprim] * M[iprim][jprim]

            Ns[i][j] = round(np.matmul(T.transpose(), np.matmul(Rs[i][j], T))) + 0.5

    

    np.save('transformed_matrix', DCT)

if __name__ == __main__:
    main()