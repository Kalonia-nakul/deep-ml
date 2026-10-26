import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    x = np.array(x0, dtype=float)   # current parameter values
    m = np.zeros_like(x)            # running average of gradients (first moment)
    v = np.zeros_like(x)            # running average of squared gradients (second moment)

    for t in range(1, num_iterations + 1):   # t = time step, starting at 1
        g = grad(x)                          # gradient of f at the current x

        m = beta1 * m + (1 - beta1) * g          # update first moment
        v = beta2 * v + (1 - beta2) * g**2       # update second moment

        m_hat = m / (1 - beta1**t)               # bias-corrected first moment
        v_hat = v / (1 - beta2**t)               # bias-corrected second moment

        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x


def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])
