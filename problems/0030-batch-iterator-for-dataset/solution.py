import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
    n_samples = X.shape[0]  # total number of rows in X
    batches = []

    for start in range(0, n_samples, batch_size):  # start = first row index of this batch
        end = min(start + batch_size, n_samples)   # end = one past the last row, capped at n_samples
        if y is not None:
            batches.append([X[start:end], y[start:end]])
        else:
            batches.append(X[start:end])

    return batches