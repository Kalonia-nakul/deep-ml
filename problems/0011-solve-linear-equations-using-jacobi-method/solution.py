import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    size = len(b)

    x = np.zeros(size)  # initial guess, all zeros

    for _ in range(n):
        x_new = np.zeros(size)
        for i in range(size):
            s = sum(A[i][j] * x[j] for j in range(size) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        x = x_new  # full precision carried forward, no rounding here

    return [round(val, 4) for val in x]  # round only at the very end