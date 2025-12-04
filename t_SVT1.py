import numpy as np
from higher_order_gsvd_two_matrices import higher_order_gsvd_two_matrices
from compute_significance import compute_significance

def t_SVT1(bbb, rho, p, m1, m2):
    # no-fill
    C1 = bbb[:m1, :, 0]
    C2 = bbb[:, :, 1]

    epsilon = 1e-5
    tsize = bbb.shape
    Tensor_L = np.zeros(tsize)

    tempU, tempU2, temps, temps2, tempV = higher_order_gsvd_two_matrices(C1, C2)
    (S1, S2, U1, U2, V, original_indices,idx, F, _, _) = compute_significance(tempU, tempU2, temps, temps2, tempV)


    # Create an extended U1 matrix
    U1_extended = np.zeros((m2, U1.shape[1]))
    U1_extended[:m1, :] = U1
    U1 = U1_extended

    # Process S1 threshold
    mask = S1 > rho
    C = S1.copy()
    C[mask] = C[mask] - 1 / ((C[mask] - epsilon) ** (1 - p))
    CC = np.maximum(C, 0)

    if V.ndim == 1:
        V = V.reshape(-1, 1)
    Tensor_L[:, :, 0] = U1 @ CC @ V.T

    # Process S2 threshold
    mask2 = S2 > rho
    C2 = S2.copy()
    C2[mask2] = C2[mask2] - 1 / ((C2[mask2] - epsilon) ** (1 - p))
    CC2 = np.maximum(C2, 0)
    Tensor_L[:, :, 1] = U2 @ CC2 @ V.T

    return U1, U2, V, S1, S2, Tensor_L, original_indices, idx, F, tempV

