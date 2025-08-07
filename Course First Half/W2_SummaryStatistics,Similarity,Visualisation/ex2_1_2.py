# exercise 2.1.2
import numpy as np

x = np.array([-0.68, -2.11, 2.39, 0.26, 1.46, 1.33, 1.03, -0.41, -0.33, 0.47])

# Compute values
mean_x = x.mean()
unbiased_std_x = x.std(ddof=1) # ddof: Delta Degrees of freedom set to 1 to make it unbiased
biased_std_x = x.std()
median_x = np.median(x)
range_x = x.max() - x.min()

# Display results
print("Vector:", x)
print("Mean:", mean_x)
print("Standard Deviation (unbiased):", unbiased_std_x)
print("Standard Deviation (biased):", biased_std_x)
print("Median:", median_x)
print("Range:", range_x)

print("Ran Exercise 2.1.1")