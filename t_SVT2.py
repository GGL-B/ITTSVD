import numpy as np
from higher_order_gsvd_two_matrices import higher_order_gsvd_two_matrices

def t_SVT2(bbb, rho):
    # fill
    C1 = bbb[:, :, 0]
    C2 = bbb[:, :, 1]
    epsilon = 1e-6
    tsize = bbb.shape
    Tensor_L = np.zeros(tsize)


    U1, U2, s, s2, V = higher_order_gsvd_two_matrices(C1, C2)

    # Process S1 threshold
    mask = s > rho
    C = s.copy()
    C[mask] = C[mask] - 1 / (C[mask] - epsilon)
    CC = np.maximum(C, 0)
    Tensor_L[:, :, 0] = U1 @ CC @ V.T

    # Process S2 threshold
    mask2 = s2 > rho
    C2 = s2.copy()
    C2[mask2] = C2[mask2] - 1 / (C2[mask2] - epsilon)
    CC2 = np.maximum(C2, 0)
    Tensor_L[:, :, 1] = U2 @ CC2 @ V.T

    return U1, U2, V, s, s2, Tensor_L

