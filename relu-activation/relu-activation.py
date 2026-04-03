import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x,0)

x =[-2, -1, 0, 3]