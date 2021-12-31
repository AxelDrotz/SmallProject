import numpy as np

m = 50
n = 50

section_size = 10

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

    # We now have the blocks in cosine-space. We now need the quantization matrix.


    np.save('transformed_matrix', DCT)

if __name__ == __main__:
    main()