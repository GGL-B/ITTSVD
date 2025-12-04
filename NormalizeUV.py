import numpy as np


def NormalizeUV(U, V, NormV, Norm):
    nSmp = V.shape[0]
    mFea = U.shape[0]

    if Norm == 2:
        if NormV:
            # Calculate the 2-norm of each column
            norms = np.sqrt(np.sum(V**2, axis=0))
            norms = np.maximum(norms, 1e-10)
            V = V / np.tile(norms, (nSmp, 1))
            U = U * np.tile(norms, (mFea, 1))
        else:
            norms = np.sqrt(np.sum(U**2, axis=0))
            norms = np.maximum(norms, 1e-10)
            V = V * np.tile(norms, (nSmp, 1))
            U = U / np.tile(norms, (mFea, 1))
    else:
        if NormV:
            norms = np.sum(np.abs(V), axis=0)
            norms = np.maximum(norms, 1e-10)
            V = V / np.tile(norms, (nSmp, 1))
            U = U * np.tile(norms, (mFea, 1))
        else:
            norms = np.sum(np.abs(U), axis=0)
            norms = np.maximum(norms, 1e-10)
            V = V * np.tile(norms, (nSmp, 1))
            U = U / np.tile(norms, (mFea, 1))

    return U, V


