import numpy as np
from hungarian import hungarian

def bestMap(L1, L2):
    """
    bestmap: permute labels of L2 to match L1 as good as possible
    parameter:
        L1 (np.ndarray)
        L2 (np.ndarray)
    return:
        np.ndarray
    """
    # input
    L1 = L1.flatten().astype(int)
    L2 = L2.flatten().astype(int)

    if L1.shape != L2.shape:
        raise ValueError("size(L1) must == size(L2)")

    # Get unique tags and sort them
    Label1 = np.unique(L1)#Return the same data as L1, but without repeat
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)

    # Building a correlation matrix G
    G = np.zeros((nClass, nClass), dtype=int)
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))


    c, t = hungarian(-G)
    # Generate new L2 labels based on mapping
    newL2 = np.zeros_like(L2)
    for i in range(nClass2):
        if c[i] <= len(Label1):
            newL2[L2 == Label2[i]] = Label1[c[i] - 1]


    return newL2




