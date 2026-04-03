def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    W = np.asarray(W, dtype=float)

    limit = np.sqrt(6/fan_in)
    return (W * 2.0-1.0 )*limit

W = [[0.5,0.5]]
fan_in = 2

W_he = he_initialization(W, fan_in)
print(W_he)