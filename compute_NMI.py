import numpy as np
def compute_NMI(r_labels, labels):
    """
    Calculate normalized mutual information NMI
    parameter:
        r_labels (np.ndarray)
        labels (np.ndarray)
    return:
        NMI
    """

    N = len(labels)
    temp = np.zeros(N, dtype=int)
    ids = np.unique(labels)
    for i in range(len(ids)):
        index = np.where(labels == ids[i])[0]
        temp[index] = i + 1

    labels = temp.copy()
    rows = max(r_labels)
    cols = max(labels)
    matrix = np.zeros((rows, cols))

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            set1 = np.where(r_labels == i)[0]
            set2 = np.where(labels == j)[0]
            matrix[i - 1, j - 1] = len(np.intersect1d(set1, set2))

    # Calculate the sum of rows and columns
    ro = matrix.sum(axis=0)
    co = matrix.sum(axis=1)

    t1 = matrix * N
    t2 = (np.outer(ro, co)).T
    tt = logme(t1 / t2)
    tt = -2 * tt * matrix
    pp = np.sum(ro * (np.log(ro / N))) + np.sum(co * (np.log(co / N)))

    accuracy = np.sum(tt) / pp
    return accuracy

#function
def logme(b):
    a = np.zeros_like(b)
    mask = (b != 0)
    a[mask] = np.log(b[mask])
    return a


