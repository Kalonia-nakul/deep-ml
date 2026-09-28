import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear',
                       lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    data = np.asarray(data, dtype=float)
    labels = np.asarray(labels).flatten()
    n = len(data)

    def linear(a, c):
        return np.dot(a, c)

    def rbf(a, c):
        return np.exp(-np.sum((a - c) ** 2) / (2 * sigma ** 2))

    K = linear if kernel == 'linear' else rbf

    alpha = [0.0] * n      
    b = 0.0                

    for t in range(1, iterations + 1):
        nt = 1 / (lambda_val * t)          

        for i in range(n):
            fx = 0.0                       
            for j in range(n):
                fx += alpha[j] * labels[j] * K(data[j], data[i])
            fx += b

            if labels[i] * fx < 1:        
                alpha[i] = (1 - nt * lambda_val) * alpha[i] + nt
                b = b + nt * labels[i]
            else:                         
                alpha[i] = (1 - nt * lambda_val) * alpha[i]

    return [float(a) for a in alpha], float(b)