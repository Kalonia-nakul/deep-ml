import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.array(weights, dtype=float)  # copy so the caller's weights are not modified

    n = X.shape[0]  # number of samples

    if method == 'batch':
        step = n            # one update per epoch using all samples
    elif method == 'stochastic':
        step = 1            # one update per sample
    elif method == 'mini_batch':
        step = batch_size   # one update per block of consecutive samples
    else:
        raise ValueError("method must be 'batch', 'stochastic' or 'mini_batch'")

    for _ in range(n_epochs):
        for start in range(0, n, step):
            Xb = X[start:start + step]   # samples in this batch, original order
            yb = y[start:start + step]
            error = Xb @ w - yb          # prediction minus target, per sample
            grad = 2 * Xb.T @ error / len(yb)  # gradient of MSE w.r.t. w
            w -= learning_rate * grad
    return w
