import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    standardized = (data - mean) / std

    # covariance matrix
    cov_matrix = np.cov(standardized, rowvar=False)

    # eigenvalues and eigenvectors (symmetric matrix -> use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # sort descending (eigh returns ascending by default)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # sign convention: first significant entry should be positive
    for i in range(eigenvectors.shape[1]):
        for val in eigenvectors[:, i]:
            if abs(val) > 1e-10:
                if val < 0:
                    eigenvectors[:, i] *= -1
                break

    # return top k components
    return eigenvectors[:, :k]
