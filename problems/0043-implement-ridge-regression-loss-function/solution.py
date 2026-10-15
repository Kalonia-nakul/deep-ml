import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    y_pred = X @ w                              # predictions, one per row of X
    mse = np.mean((y_true - y_pred) ** 2)       # Mean Squared Error
    reg = alpha * np.sum(w ** 2)                # L2 penalty on the coefficients
    return float(mse + reg)