import numpy as np


def labelA(label, c, k2):
    n = label.shape[0]  # Retrieve the number of rows for the label
    num = len(c) if isinstance(c, (list, np.ndarray)) else c.shape[0]

    # Create a C matrix
    C = np.zeros((num, k2))
    for i in range(num):
        col = c[i] - 1
        C[i, col] = 1

    I = np.eye(n - num)

    # Initialize matrix A
    A_cols = n + k2 - num
    A = np.zeros((n, A_cols))

    A[:num, :k2] = C

    # Fill in the remaining rows and add diagonal elements of the identity matrix
    for i2 in range(n - num):
        row = num + i2
        col = k2 + i2
        A[row, col] = I[i2, i2]

    return A
