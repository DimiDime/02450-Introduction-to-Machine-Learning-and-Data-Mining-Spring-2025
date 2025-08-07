# exercise 3.1.2
# (requires data structures from ex. 3.1.1)
# Imports the numpy and xlrd package, then runs the ex3_1_1 code
from ex3_1_1 import *
import matplotlib.pyplot as plt

# Data attributes to be plotted
# Chgange cols to view other dimentions
i = 3 # first col
j = 2 # second col

##
# Make a simple plot of the i'th attribute against the j'th attribute
plt.plot(X[:, i], X[:, j], "o")

##
# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = plt.figure()
plt.title("NanoNose data")

for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c 
    # Notice that the third argument of the plot() command can be used to set a plot symbol. For
    # example, the command plot(x,y,’o’) plots a scatter plot with circles.
    plt.plot(X[class_mask, i], X[class_mask, j], "o", alpha=0.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()
print("Ran Exercise 3.1.2")
