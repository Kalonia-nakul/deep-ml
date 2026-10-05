import numpy as np
import math
def adaboost_fit(X, y, n_clf):
    """X: (N, F) features, y: (N,) labels in {-1, +1}, n_clf: number of stumps."""
    X = np.asarray(X)                        # keep original dtype so threshold stays 3, not 3.0
    y = np.asarray(y)
    N, F = X.shape
    w = np.full(N, 1 / N)                    # w: sample weights, shape (N,)
    clfs = []                                # list of dicts, one per stump

    for _ in range(n_clf):
        best_err, best = np.inf, None
        for f in range(F):                   # f: feature column index
            for thr in np.unique(X[:, f]):   # thr: candidate threshold
                for pol in (1, -1):          # pol: which side predicts +1
                    pred = np.where(X[:, f] >= thr, pol, -pol)
                    err = w[pred != y].sum() # err: weighted error of this stump
                    if err < best_err:
                        best_err, best = err, (f, thr, pol, pred)

        f, thr, pol, pred = best
        alpha = 0.5 * np.log((1 - best_err) / (best_err + 1e-10))
        w *= np.exp(-alpha * y * pred)       # raise weight of misclassified samples
        w /= w.sum()

        clfs.append({
            'polarity': int(pol),
            'threshold': thr.item(),         # plain Python number (3, not np.int64(3))
            'feature_index': int(f),
            'alpha': float(alpha),
        })
    return clfs
