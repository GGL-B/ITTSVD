import numpy as np
from scipy.linalg import norm
from t_SVT1 import t_SVT1
from prox_l1 import prox_l1

def TSN1(miu, D1, D2, Tensor_X, p):
    """
       parameters:
           D1, D2: Load irregular data types and Construct matrices D1 and D2
           Tensor_X: irregular data
            miu
           Weight p

       returns:
           U, U2, V, sigma, sigma2, qq1, Tensor_epsilon, Tensor_L,
                   Tensor_Y, original_indices, idx, FI, tempV
    """


    m1 = D1.shape[0]
    m2 = D2.shape[0]
    tsize = Tensor_X.shape
    N, M, K = tsize[0], tsize[1], tsize[2]

    # regularization parameter
    lambda_ = 10 / np.sqrt(max(N, M) * K)

    # Initialize tensor
    Tensor_Y = np.zeros(tsize)
    Tensor_epsilon = np.zeros(tsize)
    Tensor_L = np.zeros(tsize)

    # parameter initialization
    pro = 1.1
    tol = 1e-1
    miumax = 1e+5
    display = 1
    qq1 = []
    maxiter=150
    k=1

    # main
    for iter in range(maxiter):
        if display:
            print(f'\n***************************** iter: {iter + 1} ******************************')

        # Save the results of the previous iteration
        L_pre = Tensor_L.copy()
        epsilon_pre = Tensor_epsilon.copy()

        bbb = np.real(Tensor_X - Tensor_epsilon - Tensor_Y / miu)

        # Irregular low-rank tensor L iteration
        U, U2, V, sigma, sigma2, Tensor_L, original_indices, idx, FI, tempV = t_SVT1(
            bbb, 1 / miu, p, m1, m2)


        #Irregular sparse tensor_epsilon iteration
        Tensor_epsilon = prox_l1(Tensor_X - Tensor_L - Tensor_Y / miu, lambda_ / miu)

        # Iterative convergence
        dY = Tensor_L + Tensor_epsilon - Tensor_X
        chgL = np.max(np.abs(L_pre - Tensor_L))
        chgS = np.max(np.abs(epsilon_pre - Tensor_epsilon))
        chg = max(chgL, chgS, np.max(np.abs(dY)))
        err = norm(dY.flatten(), 2)

        if iter % 1 == 0:
            qq1.append(err)
            k += 1

        if display:
            print(f'miu:{miu:.4e}, chg:{chg:.4e}, chgL:{chgL:.4e}, chgS:{chgS:.4e}, err:{err:.4e}')

        # Convergence judgment
        if err < tol:
            print(' !!!stopped by termination rule!!! ')
            break

        # Lagrange Multiplier Tensor Y iteration
        Tensor_Y = np.real(Tensor_Y + miu * dY)

        # Step size miu iteration
        miu = min(pro * miu, miumax)

        return (U, U2, V, sigma, sigma2, qq1, Tensor_epsilon, Tensor_L, Tensor_Y,
                original_indices, idx, FI, tempV)

