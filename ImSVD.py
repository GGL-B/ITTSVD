import numpy as np
def imsvd2(Y, rho, p, m1, m2):
    n1, n2, n3 = Y.shape
    Tensor_L = np.zeros((n1, n2, n3))

    # Apply FFT along the third dimension
    Y = np.fft.fft(Y, axis=2)
    epsilon = 1e-8

    # First frontal slice
    U, sigma, V = np.linalg.svd(Y[:, :, 0], full_matrices=False)
    U=U.real
    sigma = sigma.real
    V = V.real

    U[m1 + 1:m2, :] = 0
    mask = sigma > rho
    C = sigma.copy()
    C[mask] = C[mask] - 1 / ((C[mask] - epsilon) ** (1 - p))
    CC = np.maximum(C, 0)
    Tensor_L[:, :, 0] = U @ np.diag(CC) @ V # Reconstruct using modified singular values

    # Second frontal slice
    U2, sigma2, V2 = np.linalg.svd(Y[:, :, 1], full_matrices=False)
    V2 = V2.real
    U2 = U2.real
    sigma2 = sigma2.real

    mask2 = sigma2 > rho
    C2 = sigma2.copy()
    C2[mask2] = C2[mask2] - 1 / ((C2[mask2] - epsilon) ** (1 - p))
    CC2 = np.maximum(C2, 0)
    # Tensor_L[:, :, 1] = np.real(U2 @ np.diag(CC2) @ V2.T)
    Tensor_L[:, :, 1] = U2 @ np.diag(CC2) @ V2

    # Apply inverse FFT along the third dimension
    Tensor_L = np.fft.ifft(Tensor_L, axis=2)

    #Convert to real numbers
    Tensor_L = Tensor_L.real

    return U, U2, V, V2, sigma, sigma2, Tensor_L



