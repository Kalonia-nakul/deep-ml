import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))   
    return shifted - log_sum_exp