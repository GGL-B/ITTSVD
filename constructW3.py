import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
import time
import warnings


def EuDist2(X, Y=None, bSqrt=True):
    """
    Calculate Euclidean distance matrix
    """

    if Y is None:
        Y = X

    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T
    XY = np.dot(X, Y.T)

    D = XX + YY - 2 * XY
    D[D < 0] = 0

    if bSqrt:
        D = np.sqrt(D)

    return D


def constructW(fea, options=None):
    """
    fea: Rows of vectors of data points.
    options:  The fields in options that can be set
    """

    start_time = time.time()

    # parameter check
    if options is None:
        options = {}
    elif not isinstance(options, dict):
        raise ValueError('parameter error!')

    # Set default metrics
    if 'Metric' not in options:
        options['Metric'] = 'Cosine'
    metric_lower = options['Metric'].lower()

    if metric_lower not in ['euclidean', 'cosine']:
        raise ValueError('Metric does not exist!')

    if metric_lower == 'cosine' and 'bNormalized' not in options:
        options['bNormalized'] = 0


    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'
    neighbor_lower = options['NeighborMode'].lower()

    if neighbor_lower not in ['knn', 'supervised']:
        raise ValueError('NeighborMode does not exist!')

    if neighbor_lower == 'supervised':
        if 'bLDA' not in options:
            options['bLDA'] = 0
        if options['bLDA']:
            options['bSelfConnected'] = 1
        if 'k' not in options:
            options['k'] = 0
        if 'gnd' not in options:
            raise ValueError('Label(gnd) should be provided under "Supervised" NeighborMode!')
        if fea is not None and len(options['gnd']) != fea.shape[0]:
            raise ValueError('gnd doesn\'t match with fea!')


    if 'WeightMode' not in options:
        options['WeightMode'] = 'Binary'
    bBinary = False
    weight_lower = options['WeightMode'].lower()

    if weight_lower == 'binary':
        bBinary = True
    elif weight_lower == 'heatkernel':
        if options['Metric'].lower() != 'euclidean':
            warnings.warn('"HeatKernel" WeightMode should be used under "Euclidean" Metric!')
            options['Metric'] = 'Euclidean'
        if 't' not in options:
            options['t'] = 1
    elif weight_lower == 'cosine':
        if options['Metric'].lower() != 'cosine':
            warnings.warn('"Cosine" WeightMode should be used under "Cosine" Metric!')
            options['Metric'] = 'Cosine'
        if 'bNormalized' not in options:
            options['bNormalized'] = 0
    else:
        raise ValueError('WeightMode does not exist!')



    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 1


    tmp_T = time.time()


    # Obtain sample quantity
    if 'gnd' in options:
        nSmp = len(options['gnd'])
    else:
        nSmp = fea.shape[0]

    maxM = 62500000
    BlockSize = max(1, int(maxM / (nSmp * 3)))

    # Supervision mode processing
    if neighbor_lower == 'supervised':
        Label = np.unique(options['gnd'])
        nLabel = len(Label)

        if options['bLDA']:
            G = np.zeros((nSmp, nSmp))
            for idx in range(nLabel):
                classIdx = options['gnd'] == Label[idx]
                G[np.ix_(classIdx, classIdx)] = 1.0 / np.sum(classIdx)
            W = sp.csr_matrix(G)
            elapse = time.time() - start_time
            return W, elapse

        # Supervised learning with different weight modes
        if weight_lower == 'binary':
            if options['k'] > 0:
                G_data = []
                G_row = []
                G_col = []

                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    if len(classIdx) == 0:
                        continue

                    D = EuDist2(fea[classIdx, :], fea[classIdx, :], False)
                    idx = np.argsort(D, axis=1)[:, :options['k'] + 1]

                    for j in range(len(classIdx)):
                        for k in range(options['k'] + 1):
                            G_row.append(classIdx[j])
                            G_col.append(classIdx[idx[j, k]])
                            G_data.append(1.0)

                G = sp.coo_matrix((G_data, (G_row, G_col)), shape=(nSmp, nSmp))
                G = G.maximum(G.T)
            else:
                G = np.zeros((nSmp, nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    G[np.ix_(classIdx, classIdx)] = 1.0

            if not options['bSelfConnected']:
                G = G - sp.diags(G.diagonal())

            W = sp.csr_matrix(G)

        elif weight_lower == 'heatkernel':
            if options['k'] > 0:
                G_data = []
                G_row = []
                G_col = []

                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    if len(classIdx) == 0:
                        continue

                    D = EuDist2(fea[classIdx, :], fea[classIdx, :], True)
                    idx = np.argsort(D, axis=1)[:, :options['k'] + 1]
                    dump = np.sort(D, axis=1)[:, :options['k'] + 1]
                    dump = np.exp(-dump / (2 * options['t'] ** 2))

                    for j in range(len(classIdx)):
                        for k in range(options['k'] + 1):
                            G_row.append(classIdx[j])
                            G_col.append(classIdx[idx[j, k]])
                            G_data.append(dump[j, k])

                G = sp.coo_matrix((G_data, (G_row, G_col)), shape=(nSmp, nSmp))
            else:
                G = np.zeros((nSmp, nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    D = EuDist2(fea[classIdx, :], None, 0)
                    # D = EuDist2(fea[classIdx, :], fea[classIdx, :], True)
                    D = np.exp(-D / (2 * options['t'] ** 2))
                    G[np.ix_(classIdx, classIdx)] = D

            if not options['bSelfConnected']:
                np.fill_diagonal(G, 0)

            W = G.maximum(G.T)


        elif weight_lower == 'cosine':
            # data normalization
            if not options['bNormalized']:
                if issparse(fea):
                    fea_norm = np.sqrt(np.array(fea.power(2).sum(axis=1))).flatten()
                    fea_norm[fea_norm == 0] = 1e-10
                    fea = fea.multiply(1.0 / fea_norm[:, np.newaxis])

                else:
                    fea_norm = np.sqrt(np.sum(fea ** 2, axis=1))
                    fea_norm[fea_norm == 0] = 1e-12
                    fea = fea / fea_norm[:, np.newaxis]

            if options['k'] > 0:
                G_data = []
                G_row = []
                G_col = []

                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    if len(classIdx) == 0:
                        continue

                    D = fea[classIdx, :] @ fea[classIdx, :].T
                    idx = np.argsort(-D, axis=1)[:, :options['k'] + 1]
                    dump = -np.sort(-D, axis=1)[:, :options['k'] + 1]

                    for j in range(len(classIdx)):
                        for k in range(options['k'] + 1):
                            G_row.append(classIdx[j])
                            G_col.append(classIdx[idx[j, k]])
                            G_data.append(dump[j, k])

                G = sp.coo_matrix((G_data, (G_row, G_col)), shape=(nSmp, nSmp))
            else:
                G = np.zeros((nSmp, nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd'] == Label[i])[0]
                    G[np.ix_(classIdx, classIdx)]  = fea[classIdx, :] @ fea[classIdx, :].T

            if not options['bSelfConnected']:
                np.fill_diagonal(G, 0)

            W = G.maximum(G.T)

        elapse = time.time() - start_time
        return W, elapse

    # KNN mode processing
    if neighbor_lower == 'knn' and options.get('k', 0) > 0:
        k = options['k']

        if options['Metric'].lower() == 'euclidean':
            G_data = []
            G_row = []
            G_col = []

            for i in range(0, nSmp, BlockSize):
                end_idx = min(i + BlockSize, nSmp)
                smpIdx = np.arange(i, end_idx)

                dist = EuDist2(fea[smpIdx, :], fea, False)
                idx = np.argsort(dist, axis=1)[:, :k + 1]
                dump = np.sort(dist, axis=1)[:, :k + 1]

                if not bBinary:
                    dump = np.exp(-dump / (2 * options['t'] ** 2))

                for j in range(len(smpIdx)):
                    for m in range(k + 1):
                        G_row.append(smpIdx[j])
                        G_col.append(idx[j, m])
                        if not bBinary:
                            G_data.append(dump[j, m])
                        else:
                            G_data.append(1.0)

            W = sp.coo_matrix((G_data, (G_row, G_col)), shape=(nSmp, nSmp))

        else:  # Cosine metric
            if not options.get('bNormalized', 0):
                if issparse(fea):
                    fea_norm = np.sqrt(np.array(fea.power(2).sum(axis=1))).flatten()
                    fea_norm[fea_norm == 0] = 1e-10
                    fea = fea.multiply(1.0 / fea_norm[:, np.newaxis])
                else:
                    fea_norm = np.sqrt(np.sum(fea ** 2, axis=1))
                    fea_norm[fea_norm == 0] = 1e-12
                    fea = fea / fea_norm[:, np.newaxis]

            G_data = []
            G_row = []
            G_col = []

            for i in range(0, nSmp, BlockSize):
                end_idx = min(i + BlockSize, nSmp)
                smpIdx = np.arange(i, end_idx)

                dist = np.dot(fea[smpIdx, :], fea.T)
                idx = np.argsort(-dist, axis=1)[:, :k + 1]
                dump = -np.sort(-dist, axis=1)[:, :k + 1]

                for j in range(len(smpIdx)):
                    for m in range(k + 1):
                        G_row.append(smpIdx[j])
                        G_col.append(idx[j, m])
                        G_data.append(dump[j, m])

            W = sp.coo_matrix((G_data, (G_row, G_col)), shape=(nSmp, nSmp))

        if weight_lower == 'binary':
            W.data[:] = 1.0

        # Semi supervised learning processing
        if options.get('bSemiSupervised', False):
            tmpgnd = options['gnd'][options['semiSplit']]
            Label = np.unique(tmpgnd)
            nLabel = len(Label)

            G = np.zeros((np.sum(options['semiSplit']), np.sum(options['semiSplit'])))
            for idx in range(nLabel):
                classIdx = tmpgnd == Label[idx]
                G[np.ix_(classIdx, classIdx)] = 1.0

            Wsup = sp.csr_matrix(G)
            same_weight = options.get('SameCategoryWeight', 1)

            # Update weight matrix
            W = W.tolil()
            semi_idx = np.where(options['semiSplit'])[0]
            for i in range(len(semi_idx)):
                for j in range(len(semi_idx)):
                    if Wsup[i, j] > 0:
                        W[semi_idx[i], semi_idx[j]] = same_weight
            W = W.tocsr()


        if not options['bSelfConnected']:
            W = W - sp.diags(W.diagonal())

        W = W.maximum(W.T)
        elapse = time.time() - start_time
        return W, elapse

    # Fully connected graph processing
    if options['Metric'].lower() == 'euclidean':
        W = EuDist2(fea, None, False)
        W = np.exp(-W / (2 * options['t'] ** 2))
    else:
        if not options.get('bNormalized', 0):
            if issparse(fea):
                fea_norm = np.sqrt(np.array(fea.power(2).sum(axis=1))).flatten()
                fea_norm[fea_norm == 0] = 1e-10
                fea = fea.multiply(1.0 / fea_norm[:, np.newaxis])
            else:
                fea_norm = np.sqrt(np.sum(fea ** 2, axis=1))
                fea_norm[fea_norm == 0] = 1e-12
                fea = fea / fea_norm[:, np.newaxis]

        W = np.dot(fea, fea.T)

    if not options['bSelfConnected']:
        np.fill_diagonal(W, 0)

    W = np.maximum(W, W.T)
    elapse = time.time() - start_time
    return sp.csr_matrix(W), elapse





