import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
    X = np.asarray(X)
    y = np.asarray(y)

    if seed is not None:
        np.random.seed(seed)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]