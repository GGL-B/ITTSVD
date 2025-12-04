import numpy as np

def higher_order_gsvd_two_matrices(D1, D2):

    A1 = D1.T @ D1
    A2 = D2.T @ D2
    #Construct S
    S12 = 0.5 * (A1 @ np.linalg.pinv(A2) + A2 @ np.linalg.pinv(A1))

    # Feature decomposition and taking the real part
    _, V = np.linalg.eig(S12)
    V = np.real(V)

    # Calculate the projection matrix
    pinv_V = np.linalg.pinv(V)
    B1 = (pinv_V @ D1.T).T
    B2 = (pinv_V @ D2.T).T


    S1_diag = np.sqrt(np.sum(B1**2, axis=0))
    S1_diag[S1_diag < 1e-10] = 0
    S1 = np.diag(S1_diag)

    S2_diag = np.sqrt(np.sum(B2**2, axis=0))
    S2_diag[S2_diag < 1e-10] = 0
    S2 = np.diag(S2_diag)

    U1 = B1 @ np.linalg.pinv(S1)
    U2 = B2 @ np.linalg.pinv(S2)

    return U1, U2, S1, S2, V