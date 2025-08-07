# exercise 2.2.2
import numpy as np

from dtuimldmtools import similarity

# Generate two data objects with M random attributes
M = 5
x = np.random.rand(1, M)
y = np.random.rand(1, M)

# Two constants
a = 1.5
b = 1.5

# Check the statements in the exercise
print(
    "Cosine scaling: %.4f "
    % (similarity(x, y, "cos") - similarity(a * x, y, "cos"))[0, 0]
)
print(
    "ExtendedJaccard scaling: %.4f "
    % (similarity(x, y, "ext") - similarity(a * x, y, "ext"))[0, 0]
)
print(
    "Correlation scaling: %.4f "
    % (similarity(x, y, "cor") - similarity(a * x, y, "cor"))[0, 0]
)
print(
    "Cosine translation: %.4f "
    % (similarity(x, y, "cos") - similarity(b + x, y, "cos"))[0, 0]
)
print(
    "ExtendedJaccard translation: %.4f "
    % (similarity(x, y, "ext") - similarity(b + x, y, "ext"))[0, 0]
)
print(
    "Correlation translation: %.4f "
    % (similarity(x, y, "cor") - similarity(b + x, y, "cor"))[0, 0]
)

print(x)
print(y)

print("Ran Exercise 2.2.2")

#help(similarity)

# 2.2.3
'''
Discuss the practical implications of similarity measures that are translation and/or
scaling invariant. You can base the discusison on the image digits dataset but think
also of non-image example (e.g. retrieving documents based on the bag-of-words rep-
resentation from the previous exercise).
'''

'''
# Translation invariance: 

A similarity measure is translation invariant if it 
remains unchanged when a constant is added to all elements of the vectors 
(i.e., shifting the data).

Practical Implications:
Robustness to Shifts: Translation-invariant measures are unaffected by changes in the baseline or origin of the data. 
This is useful when the absolute values of the data are less important than the relative differences.

Applications:
Text Analysis: In document similarity, the presence or absence of words (binary or TF-IDF vectors) matters more than their absolute frequencies.
Image Processing: In image recognition, pixel intensity shifts (e.g., due to lighting changes) do not affect the similarity of image features.
Signal Processing: In time-series analysis, shifting the signal in time does not change the similarity of patterns.


# Scaling invariance

A similarity measure is scaling invariant if it remains unchanged 
when all elements of the vectors are multiplied by a constant (i.e., scaling the data).

Practical Implications:
Robustness to Magnitude: Scaling-invariant measures are unaffected
by differences in the magnitude or units of measurement. 
This is useful when the relative proportions of the data are more important than their absolute values.

Applications:
Text Analysis: In document similarity, the relative frequency of words (e.g., TF-IDF) matters more than their raw counts.
Economics: In comparing economic indicators, relative growth rates are often more meaningful than absolute values.
Biology: In gene expression analysis, the relative expression levels of genes are more important than their absolute values.
Example:

Cosine Similarity is scaling invariant because it normalizes the vectors by their magnitudes, making it insensitive to scaling.
Pearson Correlation is both translation and scaling invariant
because it standardizes the data (subtracts the mean and divides by the standard deviation).




# Combined Translation and Scaling Invariance
A similarity measure that is both translation and scaling invariant is robust to both shifts and changes in magnitude.

Practical Implications:
Universal Applicability: Such measures are widely applicable across domains because they are insensitive to both the origin and scale of the data.
Applications:
Machine Learning: In clustering or classification, 
these measures can handle data with varying baselines and scales without requiring preprocessing (e.g., normalization or standardization).
Data Integration: When combining datasets from different sources, these measures can handle differences in units or baselines.
Pattern Recognition: In identifying patterns, these measures focus on the shape or structure of the data rather than its absolute values.
Example:

Pearson Correlation is both translation and scaling invariant, 
making it a popular choice for comparing datasets with different scales or baselines.



# Correlation is both translation and scaling invariant. 



'''