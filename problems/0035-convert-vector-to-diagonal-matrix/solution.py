import numpy as np

def make_diagonal(x):
    n = len(x) 
    m = np.zeros((n, n))
    m[np.arange(n), np.arange(n)] = x
    return m