def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    W = np.asarray(W, dtype=float)
    limit = np.sqrt(6/(fan_in + fan_out))

    return ( W * 2-1) * limit

    W = [[0.0,1.0],[1.0,0.0]]
    fan_in = 2
    fan_out = 2 

    W_xavier = xavier_initialization(W, fan_in, fan_out)
    print(W_xavier)