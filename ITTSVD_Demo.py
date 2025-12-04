import numpy as np
import scipy.io as sio
from scipy import sparse
from TSN1 import TSN1
from SS import SS
import pandas as pd

"""
   ITTSVD: Irregular tensor singular value decomposition 
   Input: 
        Load irregular data types and construct irregular tensors
   Output:
        evaluation indicators:  ACC(Accuracy)
                                ARI(Adjusted Rand Index)
                                AMI(Adjusted Mutual Information)
                                NMI(Normalized Mutual Information)
"""

# init. variables
miu = 0.01

# # Loading data
df = pd.read_csv('Dataset/Sim1_D1.csv', header=None)
D1 = df.iloc[0:1717, 0:529]
df2 = pd.read_csv('Dataset/Sim1_D2.csv', header=None)
D2=df2.iloc[0:2296, 0:529]
df3 = pd.read_csv('Dataset/Sim1_labels.csv', header=None)
labels=df3.iloc[0:529, :]

# Initialize tensor and fill data
m1, n1 = D1.shape
m2 = D2.shape[0]
# Construct Tensor_X
Tensor_X = np.zeros((max(m1, m2), n1, 2))
Tensor_X[:m1, :, 0] = D1
Tensor_X[:m2, :, 1] = D2
# Obtain a unique tag and its frequency of occurrence
unique_labels, counts = np.unique(labels, return_counts=True)
num_classes = len(unique_labels)


# Initialize storage structure
C1 = [[] for _ in range(20)]
C2 = [[] for _ in range(20)]
C3 = [[] for _ in range(20)]
C4 = [[] for _ in range(20)]
C5 = [[] for _ in range(20)]
C6 = [[] for _ in range(20)]
LL1 = [[] for _ in range(20)]
LL2 = [[] for _ in range(20)]
EE1 = [[] for _ in range(20)]
EE2 = [[] for _ in range(20)]
components1 = [[] for _ in range(20)]
components2 = [[] for _ in range(20)]
rank = [[] for _ in range(20)]
chu1 = [[] for _ in range(20)]

# Initialize evaluation indicator storage
yy1 = np.zeros((20, 5))  # accuracy
uu1 = np.zeros((20, 5))  # ari
ii1 = np.zeros((20, 5))  # ami
oo1 = np.zeros((20, 5))  # nmi

ll = np.zeros((labels.shape[0], 20, 5))
pre_label = np.zeros((labels.shape[0], 20, 5))
F_final = np.zeros((num_classes,labels.shape[0],20, 5))

k = 0

#start iter
for p in np.arange(0, 1.0, 0.05):
    # Alternating iteration updates the main variables
    U, U2, V2, sigma, sigma2, qq1, Tensor_epsilon1, Tensor_L1, _, original_indices, idx, FI, tempV = TSN1(
        miu, D1, D2, Tensor_X, p)

    components1[k] = original_indices
    components2[k] = idx
    rank[k] = FI
    C1[k] = U
    C2[k] = U2
    C3[k] = tempV
    C4[k] = sigma
    C5[k] = sigma2
    C6[k] = V2
    LL1[k] = Tensor_L1[:, :, 0]
    LL2[k] = Tensor_L1[:, :, 1]
    EE1[k] = Tensor_epsilon1[:, :, 0]
    EE2[k] = Tensor_epsilon1[:, :, 1]
    chu1[k] = qq1


    j = 0
    for a in np.arange(0.2, 1.0, 0.2):
        # Construct V matrix
        V_top = Tensor_L1[:m1, :, 0]
        V_middle = Tensor_L1[:, :, 1]
        V_bottom = a * V2.T
        V = np.vstack([V_top, V_middle, V_bottom])
        V = np.real(V)

        # Filter line
        row_sums = np.sum(V, axis=1)
        rows_to_keep = row_sums >= 0.01
        c_filtered = V[rows_to_keep, :]
        c_filtered = np.maximum(c_filtered, 0)


        # Calculate evaluation indicators
        ari, ami, nmi, accuracy, l, pre, F = SS(c_filtered, labels)
        # result
        print(f"ARI: {ari}")
        print(f"AMI: {ami}")
        print(f"NMI: {nmi}")
        print(f"ACC: {accuracy}")
        yy1[k, j] = accuracy
        uu1[k, j] = ari
        ii1[k, j] = ami
        oo1[k, j] = nmi
        ll[:l.size, k, j] = l
        pre_label[:pre.size, k, j] = pre
        F_final[:F.shape[0], :F.shape[1], k, j] = F
        j += 1
    k += 1

# Calculate maximum accuracy
ac = np.max(yy1, axis=0)
acc = np.max(ac)



