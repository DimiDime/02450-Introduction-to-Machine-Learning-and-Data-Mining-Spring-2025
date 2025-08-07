import numpy as np

# Given Euclidean distances from O8 to O1...O7
distances = np.array([5.11, 4.79, 4.90, 4.74, 2.96, 5.16, 2.88])

# Parameters
N = len(distances)        # Number of reference points
M = 7                     # Dimensionality
sigma_squared = 1         # Variance (σ² = 1)
sigma = np.sqrt(sigma_squared)

# Pre-factor in the Gaussian kernel formula
prefactor = 1 / (N * ((2 * np.pi) ** (M / 2)) * (sigma ** M))

# Compute the density p(O8)
exponents = -0.5 * distances**2
density = prefactor * np.sum(np.exp(exponents))

# Output result
print(f"Estimated density at O8: {density:.2e}")
