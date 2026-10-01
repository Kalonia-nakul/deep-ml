import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X, dtype=float)   
    n_features = X.shape[1]       
    columns = []
    for d in range(degree + 1):  
        for idx in combinations_with_replacement(range(n_features), d):
            columns.append(np.prod(X[:, list(idx)], axis=1))
    out = np.column_stack(columns)   
    return np.sort(out, axis=1) 