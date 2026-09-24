import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here


    A = np.array(A, dtype=float)
    M = A.T @ A                      # symmetric 2x2: [[p, q], [q, r]]
    p, q, r = M[0, 0], M[0, 1], M[1, 1]

    # Closed-form eigenvalues of a symmetric 2x2 matrix
    mean = (p + r) / 2
    radius = np.sqrt(((p - r) / 2) ** 2 + q ** 2)
    lam1 = mean + radius             # larger eigenvalue
    lam2 = mean - radius             # smaller eigenvalue

    def eigvec(lam):
        if abs(q) > 1e-15:
            v = np.array([q, lam - p])
        else:                        # M already diagonal
            v = np.array([1.0, 0.0]) if abs(lam - p) < 1e-15 else np.array([0.0, 1.0])
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-15 else np.array([1.0, 0.0])

    v1, v2 = eigvec(lam1), eigvec(lam2)
    V = np.column_stack([v1, v2])    # V diagonalizes A^T A exactly

    B = A @ V                        # columns of B are now exactly orthogonal
    s1 = np.linalg.norm(B[:, 0])
    s2 = np.linalg.norm(B[:, 1])

    u1 = B[:, 0] / s1 if s1 > 1e-15 else np.array([1.0, 0.0])
    u2 = B[:, 1] / s2 if s2 > 1e-15 else np.array([0.0, 1.0])

    U = np.column_stack([u1, u2])
    S = np.array([s1, s2])
    Vt = V.T

    return U, S, Vt