import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    # T and S must be square to even have an inverse
    if T.shape[0] != T.shape[1] or S.shape[0] != S.shape[1]:
        return -1

    # A square matrix is invertible only if its determinant is nonzero
    if abs(np.linalg.det(T)) < 1e-9 or abs(np.linalg.det(S)) < 1e-9:
        return -1

    # Shapes must line up for T^-1 @ A @ S to be valid matrix multiplication:
    # T^-1 is (n x n), so A must have n rows; A @ S needs A's columns == S's rows
    n = T.shape[0]
    if A.shape[0] != n or A.shape[1] != S.shape[0]:
        return -1

    T_inv = np.linalg.inv(T)
    result = T_inv @ A @ S
    return result.tolist()

