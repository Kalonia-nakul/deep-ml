def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here

	import numpy as np 
    n_features = len(vectors)
    n_obs = len(vectors[0])

    # mean of each feature
    means = [sum(v) / n_obs for v in vectors]

    # build the matrix
    matrix = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            total = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_obs))
            row.append(total / (n_obs - 1))
        matrix.append(row)

    return matrix