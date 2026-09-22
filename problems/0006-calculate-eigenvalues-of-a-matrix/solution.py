def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np
	first_value = None
	second_value = None 

	x = ((matrix[0][0] + matrix[1][1]) ** 2 ) - 4 * (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
	under_root = np.sqrt(x)

	first_value = ((matrix[0][0] + matrix[1][1]) + under_root ) / 2 
	second_value = ((matrix[0][0] + matrix[1][1]) - under_root ) / 2 

	if first_value > second_value : 
		return [first_value , second_value]
	
	else  : 
		return [second_value , first_value]
