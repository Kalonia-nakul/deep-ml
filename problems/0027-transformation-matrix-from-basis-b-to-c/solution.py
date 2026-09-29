import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B_mat = np.array(B, dtype=float)   # shape (3, 3): columns are the basis vectors of B
    C_mat = np.array(C, dtype=float)   # shape (3, 3): columns are the basis vectors of C

    # A vector with coordinates x_B (in basis B) has standard coordinates B_mat @ x_B.
    # Its coordinates in basis C are x_C = C_mat^{-1} @ B_mat @ x_B.
    P = np.linalg.inv(C_mat) @ B_mat

    return np.round(P, 4).tolist()