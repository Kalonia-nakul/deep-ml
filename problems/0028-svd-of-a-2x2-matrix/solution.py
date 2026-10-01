import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.asarray(A, dtype=float)

    # Step 1: M = A^T A, symmetric; a, b, c are its entries
    M = A.T @ A
    a, b, c = M[0, 0], M[0, 1], M[1, 1]

    # Step 2: eigenvalues of M in closed form
    mean = (a + c) / 2
    rad = np.hypot((a - c) / 2, b)
    lam1 = mean + rad
    lam2 = max(mean - rad, 0.0)      # clip tiny negative rounding errors

    # Step 3: singular values
    s = np.array([np.sqrt(lam1), np.sqrt(lam2)])

    # Step 4: right singular vectors (eigenvectors of M)
    theta = 0.5 * np.arctan2(2 * b, a - c)
    v1 = np.array([np.cos(theta), np.sin(theta)])
    v2 = np.array([-np.sin(theta), np.cos(theta)])
    V = np.vstack([v1, v2])          # rows are right singular vectors

    # Step 5: left singular vectors u_i = A v_i / s_i
    eps = 1e-12
    u1 = A @ v1 / s[0] if s[0] > eps else np.array([1.0, 0.0])
    u2 = A @ v2 / s[1] if s[1] > eps else np.array([-u1[1], u1[0]])
    U = np.column_stack([u1, u2])    # columns are left singular vectors

    return U, s, V
