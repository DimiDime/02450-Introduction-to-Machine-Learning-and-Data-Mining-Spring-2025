# Exercise 2.3.7
# (requires data from exercise 2.3.1)
from ex2_3_1 import *
import matplotlib.pyplot as plt
from scipy.stats import zscore

X_standarized = zscore(X, ddof=1)

plt.figure(figsize=(12, 6))
plt.imshow(X_standarized, interpolation="none", aspect=(4.0 / N), cmap=plt.cm.gray) 
plt.xticks(range(4), attributeNames)
plt.xlabel("Attributes")
plt.ylabel("Data objects")
plt.title("Fisher's Iris data matrix")
plt.colorbar()

plt.show()

print("Ran Exercise 2.3.7")



'''

This is a heatmap visualization of the standardized (z-scored) Iris dataset where:

1. **Data Organization**:
   - Each row represents one Iris flower (a data object)
   - Each column represents one of the 4 attributes (measurements) of the Iris flowers
   - The colors represent the standardized values of these measurements

2. **Standardization**:
   - The data is standardized using `zscore()` function, which means:
   - Each feature is transformed to have:
     - Mean = 0
     - Standard deviation = 1
   - This makes all features comparable on the same scale

3. **Visual Elements**:
   - The grayscale colormap (`plt.cm.gray`) shows:
     - Darker colors = lower standardized values
     - Lighter colors = higher standardized values
   - The x-axis shows the 4 attributes (sepal length, sepal width, petal length, petal width)
   - The y-axis represents individual flowers (data objects)
   - The colorbar on the right shows the scale of the standardized values

4. **Aspect Ratio**:
   - The `aspect=(4.0/N)` parameter adjusts the shape of the plot to make it visually appealing, where N is the number of samples

This type of visualization is particularly useful for:
- Identifying patterns in the data
- Spotting outliers (extremely light or dark spots)
- Seeing how different attributes vary across samples
- Understanding the distribution of measurements after standardization

The standardization makes it easier to compare the relative magnitudes of different features, since they're all brought to the same scale, regardless of their original units of measurement.

'''