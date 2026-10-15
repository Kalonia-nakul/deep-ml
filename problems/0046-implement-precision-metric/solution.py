import numpy as np
def precision(y_true, y_pred):
	# Your code here
    tp = np.sum((y_true == 1) & (y_pred == 1))   # true positives
    fp = np.sum((y_true == 0) & (y_pred == 1))   # false positives
    if tp + fp == 0:
        return 0.0                               # no positive predictions made
    return float(tp / (tp + fp))