import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.asarray(x, dtype = float)
    n = len(x)
    
    mean_x = np.mean(x)
    
    var = np.sum((x - mean_x) ** 2) / (n - 1)
    
    dev = np.sqrt(var)
    
    return var, dev

x=[1,2,3]