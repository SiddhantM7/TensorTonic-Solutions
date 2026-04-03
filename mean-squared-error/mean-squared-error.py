import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true-y_pred)**2)

    y_pred = [2,3]
    y_true = [1,1]

    print(mean_squared_error(y_pred, y_true))
    
    
