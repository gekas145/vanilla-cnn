import time
import numba
import numpy as np
from numba import jit

# Activations
def linear(x):
    return x

def linear_der(x):
    return 1.0


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0.0, x)

def relu_der(x):
    der = x > 0
    return der.astype(np.float64)


def softmax(x):
    output = np.exp(x)
    return output/np.sum(output, axis=-1)



# Layers ops
def flatten_forward(input):
    return input.reshape(input.shape[0], -1)

def flatten_backward(output, input_shape):
    return output.reshape(input_shape)

def dense_forward(input, weights, biases):
    return input @ weights.T + biases

def dense_backward(output_der, input, weights):
    B = input.shape[0]

    input_der = output_der @ weights
    biases_der = output_der.sum(axis=0)
    weights_der = np.zeros_like(weights, dtype=np.float64)
    for b in range(B):
        weights_der += np.outer(output_der[b, ...], input[b, ...])
    
    return input_der, weights_der, biases_der

def maxpool_forward(input, N):
    B = input.shape[0]
    I = input.shape[1] // N
    C = input.shape[3]
    indexes = np.zeros((B, I**2, C, 2), dtype=np.int32)
    output = np.zeros((B, I, I, C), dtype=np.float64)

    for s1 in range(I):
        for s2 in range(I):
            maxpool_window = input[:, N*s1:N*(s1+1), N*s2:N*(s2+1), :]
            maxpool_window = maxpool_window.reshape(B, N**2, C)
            indexes_window = np.argmax(maxpool_window, axis=1)
            i = indexes_window // N + N*s1
            j = indexes_window % N + N*s2
            indexes[:, I*s1 + s2, ...] = np.stack((i, j), axis=2)

            values_window = np.amax(maxpool_window, axis=1)
            output[:, s1, s2, :] = values_window

    return output, indexes

def maxpool_backward(output_der, input_shape, indexes):
    B = input_shape[0]
    C = input_shape[3]

    input_der = np.zeros(input_shape, dtype=np.float64)
    output_der_reshaped = output_der.reshape(B, -1, C)

    for b in range(B):
        for c in range(C):
            input_der[b, indexes[b, :, c, 0], indexes[b, :, c, 1], c] = output_der_reshaped[b, :, c]

    return input_der

@jit(nopython=True)
def cnn_forward(input, kernels, biases):
    B = input.shape[0]
    I = input.shape[1]
    C = kernels.shape[0]
    K = kernels.shape[1]
    output = np.zeros(shape=(B, I-K+1, I-K+1, C))

    for b in range(B):
        for c in range(C):
            for s1 in range(0, I-K+1):
                for s2 in range(0, I-K+1):
                    output[b, s1, s2, c] = np.sum(kernels[c, ...] * input[b, s1:s1+K, s2:s2+K, :]) + biases[c]
    
    return output

@jit(nopython=True)
def cnn_backward(output_der, input, kernels):
    B = input.shape[0]
    I = input.shape[1]
    C = kernels.shape[0]
    K = kernels.shape[1]

    kernels_der = np.zeros_like(kernels, dtype=numba.float64)
    input_der = np.zeros_like(input, dtype=numba.float64)
    biases_der = output_der.reshape(-1, C).sum(axis=0)

    for b in range(B):
        for c in range(C):
            for s1 in range(0, I-K+1):
                for s2 in range(0, I-K+1):
                     kernels_der[c, ...] += output_der[b, s1, s2, c] * input[b, s1:s1+K, s2:s2+K, :]
                     input_der[b, s1:s1+K, s2:s2+K, :] += output_der[b, s1, s2, c] * kernels[c, ...]

    return input_der, kernels_der, biases_der






if __name__ == "__main__":
    B = 32
    I = 28
    C1 = 3
    cnn_input = np.random.normal(loc=0., scale=1.0, size=(B, I, I, C1))
    flatten_input = np.random.normal(loc=0., scale=1.0, size=(B, I, I, C1))

    H = 640
    dense_input = np.random.normal(loc=0.0, scale=1.0, size=(B, H))


    K = 3
    C2 = 32
    cnn_kernels = np.random.normal(loc=0., scale=1.0, size=(C2, K, K, C1))
    cnn_biases = np.random.normal(loc=0., scale=1.0, size=C2)
    cnn_output_der = np.random.normal(loc=0.0, scale=1.0, size=(B, I-K+1, I-K+1, C2))

    H1 = 100
    dense_weights = np.random.normal(loc=0.0, scale=1.0, size=(H1, H))
    dense_biases = np.random.normal(loc=0.0, scale=1.0, size=(H1))
    dense_output_der = np.random.normal(loc=0.0, scale=1.0, size=(B, H1))

    maxpool_N = 2
    maxpool_input = np.random.normal(loc=0.0, scale=1.0, size=(B, I, I, C2))

    # CNN
    for i in range(2):
        start = time.time()
        cnn_output = cnn_forward(cnn_input, cnn_kernels, cnn_biases)
        cnn_input_der, cnn_kernels_der, cnn_biases_der = cnn_backward(cnn_output_der, cnn_input, cnn_kernels)

        if i == 0:
            continue

        print(f"CNN Elapsed time: {time.time() - start} seconds")
        assert cnn_output.shape == cnn_output_der.shape
        assert cnn_input.shape == cnn_input_der.shape
        assert cnn_kernels.shape == cnn_kernels_der.shape
        assert cnn_biases.shape == cnn_biases_der.shape
    
    # Dense     
    start = time.time()
    dense_output = dense_forward(dense_input, dense_weights, dense_biases)
    dense_input_der, dense_weights_der, dense_biases_der = dense_backward(dense_output_der, dense_input, dense_weights)

    print(f"Dense Elapsed time: {time.time() - start} seconds")
    assert dense_output.shape == dense_output_der.shape
    assert dense_input.shape == dense_input_der.shape
    assert dense_weights.shape == dense_weights_der.shape
    assert dense_biases.shape == dense_biases_der.shape

    # Flatten
    flatten_output = flatten_forward(flatten_input)
    flatten_input2 = flatten_backward(flatten_output, flatten_input.shape)
    assert flatten_output.shape == (flatten_input.shape[0], np.prod(flatten_input.shape[1:]))
    assert np.allclose(flatten_input, flatten_input2)

    # Maxpool
    start = time.time()
    maxpool_output, maxpool_indexes = maxpool_forward(maxpool_input, maxpool_N)
    maxpool_input_der = maxpool_backward(maxpool_output, maxpool_input.shape, maxpool_indexes)

    print(f"Maxpool Elapsed time: {time.time() - start} seconds")
    assert maxpool_output.shape == (B, I//maxpool_N, I//maxpool_N, C2)
    assert maxpool_indexes.shape == (B, (I//maxpool_N)**2, C2, 2)
    assert maxpool_input.shape == maxpool_input_der.shape















