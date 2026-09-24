import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape

    # Initialize weights to zero
    theta = np.zeros(n)

    for _ in range(iterations):
        # Predictions
        predictions = X @ theta

        # Error
        error = predictions - y

        # Gradient of (1/2m) * sum(error^2)
        gradient = (1 / m) * (X.T @ error)

        # Update weights
        theta = theta - alpha * gradient

    return theta