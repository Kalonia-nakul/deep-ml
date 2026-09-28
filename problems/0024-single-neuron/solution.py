import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here

	# l contains y predicted
	l = []
	for i in range(len(features)):
		temp = 0
		for j in range(len(features[i])):
			temp = temp + features[i][j] * weights[j]
		temp = temp + bias 
		l.append(temp)
	
	MSE = 0

	


	# probablities 

	for i in range(len(l)):
		l[i] = 1 / (1 + np.exp(-l[i]))

	l = [round(g, 4) for g in l]




	# MSE
	for i in range(len(l)):
		MSE = MSE + (labels[i] - l[i])**2

	MSE = MSE / len(labels)

	return l, MSE