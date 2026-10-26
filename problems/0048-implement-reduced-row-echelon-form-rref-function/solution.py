import numpy as np


def rref(matrix, tol=1e-10):
    # Work on a float copy so the input isn't modified and division works
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    r = 0  # index of the row where the next pivot will be placed

    for c in range(cols):          # c = current leading column
        if r >= rows:
            break

        # Step 2: pick the row (at or below r) with the largest absolute value in column c.
        # Choosing the largest entry (partial pivoting) improves numerical stability.
        pivot_row = r + np.argmax(np.abs(A[r:, c]))

        # If the whole column below r is (numerically) zero, there's no pivot here
        if abs(A[pivot_row, c]) < tol:
            A[r:, c] = 0.0
            continue

        # Swap the pivot row into position r
        if pivot_row != r:
            A[[r, pivot_row]] = A[[pivot_row, r]]

        # Step 3: scale the pivot row so the leading entry becomes 1
        A[r] = A[r] / A[r, c]

        # Step 4: eliminate column c in every other row (above and below)
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, c] * A[r]

        r += 1  # Step 5: next pivot goes in the next row

    # Clean up tiny floating-point noise (and avoid "-0.")
    A[np.abs(A) < tol] = 0.0
    return A

