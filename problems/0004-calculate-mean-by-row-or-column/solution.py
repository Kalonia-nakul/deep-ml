def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	mean = []
	if mode == "row" : 
		for i in range(len(matrix)) : 
			temp_sum = 0
			temp_mean = 0
			for j in matrix[i] :
				temp_sum = temp_sum + j
			temp_mean = temp_sum / len(matrix[i])
			mean.append(temp_mean)
		return mean

	else : 
		for i in range(len(matrix[0])):
			temp_sum = 0
			temp_mean = 0
			for j in range(len(matrix)) : 
				temp_sum = temp_sum + matrix[j][i]
			temp_mean = temp_sum / len(matrix)
			mean.append(temp_mean)
		return mean
				