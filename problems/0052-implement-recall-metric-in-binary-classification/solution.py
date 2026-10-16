import numpy as np

def recall(y_true, y_pred):
    y_true = np.asarray(y_true)  # true labels (0 or 1)
    y_pred = np.asarray(y_pred)  # model's predicted labels (0 or 1)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # actual 1, predicted 1
    fn = np.sum((y_true == 1) & (y_pred == 0))  # actual 1, predicted 0

    denominator = tp + fn  # total number of actual positives
    if denominator == 0:
        return 0.0  # no positives in y_true, avoid division by zero
    return float(tp / denominator)
