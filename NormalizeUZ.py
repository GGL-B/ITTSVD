import numpy as np


def NormalizeUZ(U, Z, NormV, Norm):

    if Norm == 2:
        if NormV:
            norms = np.sqrt(np.sum(Z**2, axis=0))
            norms = np.maximum(norms, 1e-10)
            Z = Z / norms[np.newaxis, :]
            U = U * norms[np.newaxis, :]
        else:
            norms = np.sqrt(np.sum(U**2, axis=0))
            norms = np.maximum(norms, 1e-10)
            U = U / norms[np.newaxis, :]
            Z = Z * norms[np.newaxis, :]
    else:
        if NormV:
            norms = np.sum(np.abs(Z), axis=0)
            norms = np.maximum(norms, 1e-10)
            Z = Z / norms[np.newaxis, :]
            U = U * norms[np.newaxis, :]
        else:
            norms = np.sum(np.abs(U), axis=0)
            norms = np.maximum(norms, 1e-10)
            U = U / norms[np.newaxis, :]
            Z = Z * norms[np.newaxis, :]

    return U, Z


