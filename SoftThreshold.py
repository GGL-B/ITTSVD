import numpy as np

def SoftThreshold(A, gamma):

    win1 = (A > 0)
    win2 = (A < 0)
    win3 = ((gamma - 2 * np.abs(A)) < 0)
    S = (A - 0.5*gamma) *win1 *win3 + (A + 0.5 * gamma) * win2 * win3
    return S
