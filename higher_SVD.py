import numpy as np
from scipy.linalg import eig,svd, pinv

def higher_svd (D1, D2, KK):
    Z = D2.shape[1]

    # Construct S
    A = np.zeros((D2.shape[0], D2.shape[1], 2))
    A[:,:,0] = np.dot(D1.T, D1)
    A[:,:,1] = np.dot(D2.T, D2)
    S = 0.5 * (np.dot(A[:,:,0], pinv(A[:,:,1])) + np.dot(A[:,:,1], pinv(A[:,:,0])))

    _,V = np.linalg.eig(S)
    B1 = np.dot(pinv(V), D1.T)
    B2 = np.dot(pinv(V), D2.T)

    B1 = B1.T
    B2 = B2.T

    c = np.zeros(Z)
    sigma = np.zeros((Z, Z))

    for i in range(Z):
        if KK == 1:
            c[i] = np.linalg.norm(B1[:, i])
        else:
            c[i] = np.linalg.norm(B2[:, i])
        sigma[i, i] = c[i]

    if KK == 1:
        U = np.dot(B1, pinv(sigma))
        eps = np.dot(np.dot(U, sigma), V) - D1
        U1, S1, V1 = np.linalg.svd(D1, full_matrices=False)
    else:
        U = np.dot(B2, pinv(sigma))
        # eps = np.dot(U, np.dot(sigma, V.conj().T)) - D2
        eps = np.dot(np.dot(U,sigma), V) - D2
        # U1, S1, V1 = svd(D2, full_matrices=False)
        U1, S1, V1 = np.linalg.svd(D2, full_matrices=False)

    eps2 = (U1 @ np.diag(S1) @ V1) - np.dot(U, np.dot(sigma, V))

    return S, B1, B2, V, U1, V1, S1, eps, eps2, sigma, c, U, A, Z