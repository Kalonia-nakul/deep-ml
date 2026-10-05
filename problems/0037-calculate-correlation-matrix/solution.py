import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    N = X.shape[0]                       # number of samples

    Xc = X - X.mean(axis=0)              # X with each column's mean removed
    Yc = Y - Y.mean(axis=0)              # Y with each column's mean removed

    cov = Xc.T @ Yc / N                  # (Fx, Fy): covariance of X col j with Y col k
    sx = np.sqrt((Xc ** 2).mean(axis=0)) # (Fx,): std of each X column
    sy = np.sqrt((Yc ** 2).mean(axis=0)) # (Fy,): std of each Y column

    return cov / np.outer(sx, sy)  