import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
    correct = (y_true == y_pred)
    return float(np.mean(correct))