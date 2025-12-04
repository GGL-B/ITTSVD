import numpy as np
def prox_l1(b, lambda_):
    """
    Soft threshold
    parameter:
        b:
        lambda_

    return:
        Values/matrices processed by soft thresholding
    """
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

