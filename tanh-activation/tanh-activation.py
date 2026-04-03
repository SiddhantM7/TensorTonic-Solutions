import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=float)
    return np.tanh(x)
    
    x = [0,1,-1,3]