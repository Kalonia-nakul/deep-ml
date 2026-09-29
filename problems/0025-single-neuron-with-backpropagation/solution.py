import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
    X = np.array(features, dtype=float)      
    y = np.array(labels, dtype=float)      
    w = np.array(initial_weights, dtype=float) 
    b = float(initial_bias)
    n = len(y)

    mse_values = []
    for _ in range(epochs):
       
        z = X @ w + b                         
        p = 1.0 / (1.0 + np.exp(-z))          
       
        mse_values.append(round(float(np.mean((p - y) ** 2)), 4))

        dz = (2.0 / n) * (p - y) * p * (1.0 - p)
        grad_w = X.T @ dz                     
        grad_b = np.sum(dz)                   

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    updated_weights = [round(float(v), 4) for v in w]
    updated_bias = round(float(b), 4)
    return updated_weights, updated_bias, mse_values