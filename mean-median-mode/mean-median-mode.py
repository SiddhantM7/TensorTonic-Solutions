import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.asarray(x, dtype = float)

    mean = np.mean(x)
    median = np.median(x)

    # mode

    counts = Counter(x)
    mode = counts.most_common(1)[0][0]
    return mean, median, mode 

x = [1,1,2,3]
    