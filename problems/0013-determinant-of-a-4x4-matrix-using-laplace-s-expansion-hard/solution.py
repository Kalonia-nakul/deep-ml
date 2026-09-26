def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here

	def get_minor(matrix , row , column) :
		l = []
		for i in range(len(matrix)):
			temp = []
			if i == row :
				continue
			for j in range(len(matrix)):
				if j== column : 
					continue 
				temp.append(matrix[i][j])
			l.append(temp)
		return determinant(l)
				

	def determinant(matrix) :
		x = 0
		row = 0
		if len(matrix) == 2 : 
			return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
		for col in range(len(matrix)):
			x = x + ((-1)**(1 + col))  * (matrix[row][col]) * (get_minor(matrix , row , col))

		return x
				

	return determinant(matrix)
