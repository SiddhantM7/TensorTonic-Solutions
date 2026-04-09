import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    return float(np.sqrt(np.sum((x - y) ** 2)))
    x = [1,2,3]
    y = [4,5,6]

    