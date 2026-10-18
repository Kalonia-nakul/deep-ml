import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_matrix = np.asarray(input_matrix)
    kernel = np.asarray(kernel)

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded = np.pad(input_matrix, padding, mode='constant', constant_values=0)
    padded_height, padded_width = padded.shape

    out_h = (padded_height - kernel_height) // stride + 1
    out_w = (padded_width - kernel_width) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r = i * stride  # top row of the window in the padded input
            c = j * stride  # left column of the window in the padded input
            window = padded[r:r + kernel_height, c:c + kernel_width]
            output[i, j] = np.sum(window * kernel)
    return output