import numpy as np


def compute_significance(tempU, tempU2, temps, temps2, tempV):
    epsilon = 1e-4

    # Extract diagonal elements
    sigma1 = np.diag(temps)
    sigma2 = np.diag(temps2)

    # Filter non-zero items
    non_zero_mask = (sigma1 > 1e-6) & (sigma2 > 1e-6)
    sigma1_nz = sigma1[non_zero_mask]
    sigma2_nz = sigma2[non_zero_mask]

    # Calculate logarithmic product normalization
    log_products1 = np.log(1 + sigma1 * sigma2)
    log_products = np.log(1 + sigma1_nz * sigma2_nz)
    total_log_sum = np.sum(log_products)
    p = log_products1 / (total_log_sum )

    # Calculate the deviation of the ratio
    ratio_deviations = np.abs(sigma1 - sigma2) / (sigma1 + sigma2 + epsilon)

    # Min-Max
    z = (ratio_deviations - np.min(ratio_deviations)) / (
            np.max(ratio_deviations) - np.min(ratio_deviations) )

    F = p * (1 + z)

    #Sort in descending order and calculate cumulative contribution rate
    sorted_indices = np.argsort(F)[::-1]
    sorted_F = F[sorted_indices]
    cumulative_contribution = np.cumsum(sorted_F) / (np.sum(sorted_F))

    k = np.argmax(cumulative_contribution >= 0.95) + 1

    # Truncated matrix
    original_indices = sorted_indices[:k]

    U1 = tempU[:, original_indices]
    U2 = tempU2[:, original_indices]
    S1 = temps[original_indices, :][:, original_indices]
    S2 = temps2[original_indices, :][:, original_indices]
    V = tempV[:, original_indices]

    return S1, S2, U1, U2, V, original_indices, k, F, p, z