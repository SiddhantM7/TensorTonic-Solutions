import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float) 
    beta = np.asarray(beta, dtype=float)   

    # 1. Determine axes and reshape parameters based on input dimensions
    if x.ndim == 2:
        # For (N, D), normalize over the batch dimension (N)
        axis = 0
        # Reshape (D,) to (1, D) for broadcasting
        gamma = gamma.reshape(1, -1)
        beta = beta.reshape(1, -1)
        
    elif x.ndim == 4:
        # For (N, C, H, W), normalize over N, H, W per channel C
        axis = (0, 2, 3)
        # Reshape (C,) to (1, C, 1, 1) for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        
    else:
        raise ValueError("Input must be exactly 2D or 4D")

    # 2. Compute mean and variance with keepdims=True for proper broadcasting
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    
    # 3. Normalize (subtract mean, divide by standard deviation + eps for stability)
    x_hat = (x - mu) / np.sqrt(var + eps)
    
    # 4. Scale and Shift
    y = gamma * x_hat + beta
    
    return y


x_2d = np.array([[1, 2], [3, 6], [5, 10]])
gamma_2d = np.array([1, 0.5])
beta_2d = np.array([0, 1])


print(batch_norm_forward(x_2d, gamma_2d, beta_2d))