import numpy as np

'''
Program for calculating the Adjusted Mutual Information (AMI) between two clusterings
Input: a contingency table T OR cluster label of the two clusterings in two vectors
Output: AMI: adjusted mutual information  (AMI normalized by Sqrt(HA,HB))
'''
def AMI(true_mem, mem=None):
    if mem is None:
        T = true_mem  # contingency table pre-supplied
    else:
        # build the contingency table from membership arrays
        r = np.max(true_mem)
        c = np.max(mem)

        # identify & remove the missing labels
        list_t = np.isin(np.arange(1, r + 1), true_mem)
        list_m = np.isin(np.arange(1, c + 1), mem)
        T = Contingency(true_mem, mem)
        T = T[list_t, :][:, list_m]

    r, c = T.shape
    if c == 1 or r == 1:
        raise ValueError('Clusterings should have at least 2 clusters')

    N = np.sum(T)  # total number of records

    # update the true dimensions
    a = np.sum(T, axis=1)
    b = np.sum(T, axis=0)

    # calculating the Entropies
    Ha = -np.sum((a[a != 0] / N) * np.log2(a[a != 0] / N))
    Hb = -np.sum((b[b != 0] / N) * np.log2(b[b != 0] / N))

    # calculate the MI
    MI = 0
    for i in range(r):
        for j in range(c):
            if T[i, j] > 0:
                MI += T[i, j] * np.log2(T[i, j] * N / (a[i] * b[j]))
    MI /= N

    #correcting for agreement by chanc
    AB = np.outer(a, b)
    bound = np.zeros((r, c))

    E3 = (AB / N ** 2) * np.log2(AB / N ** 2)
    E3[np.isnan(E3)] = 0

    EPLNP = np.zeros((r, c))
    log2Nij = np.log2(np.arange(1, min(np.max(a), np.max(b)) + 1) / N)

    for i in range(r):
        for j in range(c):
            nij_start = max(1, a[i] + b[j] - N)
            X = np.array([nij_start, N - a[i] - b[j] + nij_start])
            X.sort()

            if N - b[j] > X[1]:
                nom = np.concatenate([np.arange(a[i] - nij_start + 1, a[i] + 1),
                                      np.arange(b[j] - nij_start + 1, b[j] + 1),
                                      np.arange(X[1] + 1, N - b[j] + 1)])
                dem = np.concatenate([np.arange(N - a[i] + 1, N + 1),
                                      np.arange(1, X[0] + 1)])
            else:
                nom = np.concatenate([np.arange(a[i] - nij_start + 1, a[i] + 1),
                                      np.arange(b[j] - nij_start + 1, b[j] + 1)])
                dem = np.concatenate([np.arange(N - a[i] + 1, N + 1),
                                      np.arange(N - b[j] + 1, X[1] + 1),
                                      np.arange(1, X[0] + 1)])

            p1 = np.prod(nom / dem) / N

            for nij in range(nij_start, min(a[i], b[j]) + 1):
                EPLNP[i, j] += nij * log2Nij[nij - 1] * p1
                if nij < min(a[i], b[j]):
                    p1 = p1 * (a[i] - nij) * (b[j] - nij) / (nij + 1) / (N - a[i] - b[j] + nij + 1)

            CC = N * (a[i] - 1) * (b[j] - 1) / (a[i] * b[j] * (N - 1)) + N / (a[i] * b[j])
            bound[i, j] = a[i] * b[j] / N ** 2 * np.log2(CC)

    EMI_bound = np.sum(bound)
    EMI = np.sum(EPLNP - E3)

    AMI_ = (MI - EMI) / (np.sqrt(Ha * Hb) - EMI)
    NMI = MI / np.sqrt(Ha * Hb)

    # If expected mutual information negligible, use NMI
    if abs(EMI) > EMI_bound:
        print(f'The EMI is small: EMI < {EMI_bound}, setting AMI=NMI')
        AMI_ = NMI

    return AMI_

#auxiliary functions
def Contingency(Mem1, Mem2):
    if len(Mem1) != len(Mem2):
        raise ValueError('Contingency: Requires two vectors of the same length')

    Cont = np.zeros((np.max(Mem1), np.max(Mem2)), dtype=int)

    for i in range(len(Mem1)):
        Cont[Mem1[i] - 1, Mem2[i] - 1] += 1  # -1 because Python is 0-indexed

    return Cont

