import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)                  # x_i - m
    log_sum_exp = np.log(np.sum(np.exp(shifted)))      # log(sum_j exp(x_j - m))
    return shifted - log_sum_exp