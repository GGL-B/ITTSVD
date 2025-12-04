
import numpy as np
from scipy.sparse import spdiags, eye as sparse_eye
from scipy.sparse.linalg import norm as sparse_norm
from SoftThreshold import SoftThreshold
from NormalizeUZ import NormalizeUZ
from NormalizeUV import NormalizeUV

from scipy.sparse import random
from scipy.sparse import diags

def soft(x, T):
    if np.sum(np.abs(T)) == 0:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0)
        y = np.sign(x) * y
    return y


def ssnmdi_model(X, A, lambda_, k1, k2, W, options, num):
    """
   SSNMDI_model.

    parameter:
        X: Data matrix
        A: Label matrix for training samples, shape (n, n+k2-num)
        lambda_: Regularization parameter for S
        k1: Rank / dimension of latent space U/Z
        k2: Number of classes
        W: Affinity matrix (n, n)
        options: Dictionary of parameters
        num: Number of training samples

    Returns:
        U_final, Z_final, B_final, F_final, S_final
    """

    # Default parameters
    differror = 1e-5
    if 'error' in options:
        differror = options['error']

    maxIter = 100
    if 'maxIter' in options:
        maxIter = options['maxIter']

    nRepeat = 1
    if 'nRepeat' in options:
        nRepeat = options['nRepeat']

    minIterOrig = 30
    if 'minIter' in options:
        minIterOrig = options['minIter']
    minIter = minIterOrig - 1

    meanFitRatio = 0.1
    if 'meanFitRatio' in options:
        meanFitRatio = options['meanFitRatio']

    alpha = 1.0
    if 'alpha' in options:
        alpha = options['alpha']


    if np.min(X) < 0:
        raise ValueError('Input should be nonnegative!')

    # Initialize U0, Z0
    m, n = X.shape
    np.random.seed(23)
    U0 = np.abs(np.random.rand(m, k1))
    Z0 = np.abs(np.random.rand(n + k2 - num, k1))

    # Make S0: random noise where X==0
    window = (X == 0).astype(float)
    dim = X.shape
    S0 = np.random.rand(*dim) * window

    # Weight matrix scaling
    nSmp = A.shape[0]
    if 'weight' in options and options['weight'].lower() == 'ncw':
        feaSum = np.sum(X, axis=1, keepdims=True)
        D_half = np.sqrt((X.T @ feaSum).flatten())
        for i in range(nSmp):
            X[:, i] /= D_half[i]

    if 'alpha_nSmp' in options and options['alpha_nSmp']:
        alpha *= nSmp
    W = alpha * W

    if hasattr(W, 'toarray'):
        W_dense = W.toarray()
    else:
        W_dense = W

    DCol = np.sum(W_dense, axis=1).flatten()
    D = spdiags(DCol, 0, W_dense.shape[0], W_dense.shape[0])
    L = D - W_dense

    if 'NormW' in options and options['NormW']:
        D_mhalf = 1.0 / np.sqrt(DCol)
        D_mhalf_diag = diags(D_mhalf)
        L = D_mhalf_diag @ L @ D_mhalf_diag
        L = np.maximum(L, L.T)

    # Initialize B0, F0
    selectInit = 1
    if 'U' not in locals():
        B0 = np.abs(np.random.rand(k1, k2))
        F0 = np.abs(np.random.rand(k2, n))
    else:
        nRepeat = 1

    # Parameter settings
    U0, Z0 = NormalizeUZ(U0, Z0, 1, 2)
    B0, F0 = NormalizeUV(B0, F0.T, 1, 2)
    F0 = F0.T

    # Initialize variables
    Uk, Zk = U0, Z0
    Bk, Fk = B0, F0
    Sk = S0
    Ek = Zk.copy()
    mFea1, nSmp1 = Zk.shape
    Tk = np.zeros((mFea1, k1))

    # Iteration settings
    iter = 0
    converged = False
    maxIter = 100
    tol1 = 1e-5
    tol2 = 1e-5

    # For convergence tracking
    er1 = []
    er2 = []
    t1 = []
    t2 = []

    while not converged and iter < maxIter:
        iter += 1
        derta = 5e+1

        # Update regularization parameters
        alpha = np.linalg.norm(X, 1) / np.linalg.norm((A @ Zk).T, 'fro')
        beta = np.linalg.norm((A @ Zk).T, 1) / np.linalg.norm(Fk, 'fro')
        E = np.eye(k1)

        # Update U
        numerator_U = (X + Sk) @ A @ Zk
        denominator_U = Uk @ Zk.T @ A.T @ A @ Zk + alpha * Uk
        Ukl = Uk * (numerator_U / (denominator_U + 1e-10))

        # Update Z
        numerator_Z = A.T @ (X + Sk).T @ Ukl + A.T @ Fk.T @ Bk.T
        denominator_Z = A.T @ A @ Zk @ (Ukl.T @ Ukl + E)
        Zkl = Zk * (numerator_Z / (denominator_Z + 1e-10))

        # Update E and T
        Ekl = soft(Zkl - Tk / derta, alpha / derta)
        Tkl = Tk + 1.618 * derta * (Ekl - Zkl)

        # Update B
        Bkl = Bk * ((Zkl.T @ A.T @ Fk.T) / (Bk @ Fk @ Fk.T + 1e-10))
        Bkl[Bkl < 0] = 0

        # Update F
        if hasattr(W, 'toarray'):
            W_dense_current = W.toarray()
        else:
            W_dense_current = W

        FF1 = Bkl.T @ Zkl.T @ A.T + beta * Fk @ W_dense_current
        FF2 = Bkl.T @ Bkl @ Fk + beta * Fk * DCol.reshape(1, -1)
        Fkl = Fk * (FF1 / (FF2 + 1e-10))
        Fkl[Fkl < 0] = 0

        # Update S
        temp = -X + Ukl @ Zkl.T @ A.T

        # calculate miu
        sum_X = np.sum(X, axis=1)
        miu_diag = sum_X / np.median(sum_X)
        miu_matrix = np.diag(miu_diag)

        # Threshold calculation
        threshold_matrix = lambda_ * np.dot(miu_matrix, np.ones_like(X))
        Skl = SoftThreshold(temp, threshold_matrix)
        Skl = Skl * window

        # Normalize B and F
        Bkl, Fkl = NormalizeUV(Bkl, Fkl.T, 1, 2)
        Fkl = Fkl.T

        Uwk, Zwk, Bwk, Fwk, Swk = Uk, Zk, Bk, Fk, Sk
        Uk, Zk, Bk, Fk, Sk = Ukl, Zkl, Bkl, Fkl, Skl

        # Compute errors
        residual1 = X - Uk @ Zk.T @ A.T
        residual2 = Zk.T @ A.T - Bk @ Fk
        er1_val = np.mean(np.abs(residual1)) / np.linalg.norm(Uk @ Zk.T @ A.T, 'fro')
        er2_val = np.mean(np.abs(residual2)) / np.linalg.norm(Bk @ Fk, 'fro')
        er1.append(er1_val)
        er2.append(er2_val)

        # Convergence check
        temp = max([np.linalg.norm(Ukl - Uwk, 'fro'), np.linalg.norm(Zkl - Zwk, 'fro'),
                    np.linalg.norm(Bkl - Bwk, 'fro'), np.linalg.norm(Fkl - Fwk, 'fro')])
        temp = temp / np.linalg.norm(X, 'fro')

        temp1 = max(np.linalg.norm(residual1, 'fro'),
                    np.linalg.norm(residual2, 'fro')) / max(
            np.linalg.norm(Bk @ Fk, 'fro'), np.linalg.norm(Uk @ Zk.T @ A.T, 'fro'))

        t1.append(temp1)
        t2.append(temp)

        print(f"iteration count {iter} temp1 {temp1:.6f} temp {temp:.6f}")

        if temp1 < tol1 and temp < tol2:
            converged = True

    # Final outputs
    U_final, Z_final = NormalizeUZ(Ukl, Zkl, 1, 2)
    B_final, F_final = NormalizeUV(Bkl, Fkl.T, 1, 2)
    F_final = F_final.T
    S_final = Skl

    return U_final, Z_final, B_final, F_final, S_final


