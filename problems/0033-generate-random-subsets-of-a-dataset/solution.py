import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = X.shape[0]  # number of rows in the dataset

    # with replacement: same size as the dataset; without: half (integer division)
    subset_size = n_samples if replacements else n_samples // 2

    subsets = []
    for _ in range(n_subsets):
        # random row indices; replace=True lets the same index repeat within one subset
        idxs = np.random.choice(n_samples, size=subset_size, replace=replacements)
        subsets.append((X[idxs].tolist(), y[idxs].tolist()))
    return subsets