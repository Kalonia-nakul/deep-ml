import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here


	a = []
	b = []
	for i in range(len(X)) :
		if X[i][feature_i] < threshold : 
			a.append(X[i])
		else :
			b.append(X[i])


	a = np.array(a)
	b = np.array(b)

	return [b , a]