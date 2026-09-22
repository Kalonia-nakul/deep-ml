import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	matrix = []
	all_element = []
	for i in range(len(a)):
		for j in range(len(a[1])):
			all_element.append(a[i][j])

	if ( (new_shape[0] * new_shape[1]) != len(all_element) ) : 
		return []

	b = 0
	for i in range(new_shape[0]):
		temp = []
		for j in range(new_shape[1]):
			temp.append(all_element[b])
			b+=1
		matrix.append(temp)

	return matrix