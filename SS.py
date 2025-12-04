import numpy as np
from labelA import labelA
from constructW3 import constructW
from bestMap import bestMap
from ARI import ARI
from AMI import AMI
from compute_NMI import compute_NMI
from SSNMDI_model import ssnmdi_model


def SS(Y3, labels):
    """
        Implement preprocessing, graph construction, model training,
        and performance evaluation related to semi supervised clustering

        parameter:
            Y3 (np.ndarray): feature matrix
            labels (np.ndarray): Real label vector

        return:
            ari (float): Adjusted Rand Index
            ami (float): Adjusted Mutual Information
            nmi (float): Normalized Mutual Information
            accuracy (float)
            ll (np.ndarray): A subset of real labels used for evaluation
            pre_label (np.ndarray): Final predicted label
            F_final (np.ndarray): Final feature matrix
        """

    X = Y3.copy()

    if hasattr(labels, 'values'):
        label = labels.values.flatten()
    elif hasattr(labels, 'to_numpy'):
        label = labels.to_numpy().flatten()
    else:
        label = labels.copy().flatten()

    k2 = len(np.unique(label))
    scala = 0.2

    # Split data into labeled and unlabeled parts
    X_1 = []
    X_2 = []
    y_1 = []
    y_2 = []
    c = []

    for labell in np.unique(label):
        #Record the position in the label that matches the label value
        cate = np.where(label == labell)[0]

        half = int(np.ceil(len(cate) * scala))# Calculate how many samples to select (rounded up)
        local = np.random.permutation(len(cate))[:half]# Randomly select half non repeating index positions
        local_lab = cate[local]# Extract the actual values from the cat based on these random indexes
        local_non = np.setdiff1d(cate, local_lab)# Find the elements in the cat that are not in the local_1ab


        X_1.append(X[:, local_lab])
        X_2.append(X[:, local_non])
        y_1.extend(label[local_lab])
        y_2.extend(label[local_non])
        c.extend(label[local_lab])

    X_1 = np.hstack(X_1)
    X_2 = np.hstack(X_2)
    X = np.hstack([X_1, X_2])
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)
    label = np.concatenate([y_1, y_2])

    c = np.array(c)
    # Build constraint matrix A
    A = labelA(label, c, k2)

    # Construct similarity matrix W
    options = {
        'Metric': 'Cosine',
        'NeighborMode': 'KNN',
        'k': 5,
        'WeightMode': 'Cosine'
    }

    # Transpose X because constructW expects (n_samples, n_features)
    W_result = constructW(X.T, options)

    #Check the return type of ConstructW and extract the weight matrix
    if isinstance(W_result, tuple):
        if len(W_result) > 0:
            W = W_result[0]
        else:
            raise ValueError("constructW returned an empty tuple")
    elif hasattr(W_result, 'shape'):
        W = W_result
    else:

        try:
            W = np.array(W_result)
        except:
            raise ValueError(f"Unable to handle the return type of ConstructW: {type(W_result)}")
    # Ensure that W is of numerical type
    # print(f"W type: {type(W)}, shape: {getattr(W, 'shape', 'No shape')}")

    # Clear and reinitialize options
    options = {}

    num = len(c)
    # Regularization parameter
    lambda_val = 2
    k1 = 500

    # main model
    U_final, Z_final, B_final, F_final, S_final = ssnmdi_model(X, A, lambda_val, k1, k2, W, options, num)

    #Clustering cell type label
    l = np.zeros(F_final.shape[1])
    for e in range(F_final.shape[1]):
        v = F_final[:, e]
        ma = np.max(v)
        idx = np.where(v == ma)[0][0]
        l[e] = idx + 1

    # Performance evaluation
    ll = label[num:]
    l = l[num:]

    newl = bestMap(ll, l)

    # Compute evaluation metrics
    # Calculating the Normalized Mutual Information (NMI)
    nmi = compute_NMI(ll, newl)
    # Calculating the Adjusted Mutual Information (AMI)
    ami = AMI(ll, newl)
    # Calculating the Adjusted Rand Index (ARI)
    ari = ARI(ll, np.max(ll), newl, np.max(newl))
    pre_label = newl

    # Accuracy
    if len(ll) > 0:
        exact = np.where(pre_label == ll)[0]
        accuracy = len(exact) / len(newl)
    else:
        accuracy = None

    return ari, ami, nmi, accuracy, ll, newl, F_final


