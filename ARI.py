import numpy as np
from math import comb


def ARI(Clustering1, k1, Clustering2, k2):
    """
    Adjusted Rand Index
    parameter：
    Clustering1 : list/np.array
    k1 : int
    Clustering2 : list/np.array
    k2 : int
    return:
    AR : float
    """
    # input validation
    Clustering1 = np.asarray(Clustering1)
    Clustering2 = np.asarray(Clustering2)
    N = Clustering1.shape[0]

    if len(Clustering2) != N:
        raise ValueError("Clustering results must have the same length")

    # Initialize the contingency matrix
    contig_matrix = np.zeros((k1, k2), dtype=int)

    for point in range(N):
        i = Clustering1[point] - 1
        j = Clustering2[point] - 1
        if i >= k1 or j >= k2 or i < 0 or j < 0:
            raise ValueError("Cluster labels exceed specified cluster numbers")
        contig_matrix[i, j] += 1

    # Calculate row and column sums
    a = contig_matrix.sum(axis=1)
    b = contig_matrix.sum(axis=0)

    # Calculate the sum of combinations
    SumCombnij = 0
    for i in range(k1):
        for j in range(k2):
            if contig_matrix[i, j] > 1:
                SumCombnij += comb(contig_matrix[i, j], 2)

    SumCombai = 0
    for i in range(k1):
        if a[i] > 1:
            SumCombai += comb(a[i], 2)

    SumCombbj = 0
    for j in range(k2):
        if b[j] > 1:
            SumCombbj += comb(b[j], 2)

    nCh2 = comb(N, 2)
    temp = (SumCombai * SumCombbj) / nCh2

    # Dealing with situations where the denominator is zero
    denominator = 0.5 * (SumCombai + SumCombbj) - temp
    if denominator == 0:
        numerator = SumCombnij - temp
        if numerator == 0:
            return 1.0
        else:
            return 0.0

    AR = (SumCombnij - temp) / denominator
    return AR

