import time
import numba
import numpy as np
from numba import jit


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

def activation(x):
    return 1/(1 + np.exp(-x))

def activation_der(x):
    return 1/(1 + np.exp(-x))

@jit(nopython=True)
def flatten_forward(input):
    return input.reshape(input.shape[0], -1)

@jit(nopython=True)
def flatten_backward(output, input_shape):
    return output.reshape(input_shape)

@jit(nopython=True)
def dense_forward(input, weights, biases):
    return input @ weights.T + biases

@jit(nopython=True)
def dense_backward(output_der, input, weights):
    B = input.shape[0]

    input_der = output_der @ weights
    biases_der = output_der.sum(axis=0)
    weights_der = np.zeros_like(weights, dtype=np.float32)
    for b in range(B):
        weights_der += np.outer(output_der[b, ...], input[b, ...])
    
    B = float(B)
    weights_der = weights_der/B
    biases_der = biases_der/B
    
    return input_der, weights_der, biases_der

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

    kernels_der = np.zeros_like(kernels, dtype=numba.float32)
    input_der = np.zeros_like(input, dtype=numba.float32)
    biases_der = output_der.reshape(-1, C2).sum(axis=0)

    for b in range(B):
        for c in range(C):
            for s1 in range(0, I-K+1):
                for s2 in range(0, I-K+1):
                     kernels_der[c, ...] += output_der[b, s1, s2, c] * input[b, s1:s1+K, s2:s2+K, :]
                     input_der[b, s1:s1+K, s2:s2+K, :] += output_der[b, s1, s2, c] * kernels[c, ...]

    B = float(B)
    kernels_der = kernels_der/B
    biases_der = biases_der/B
    return input_der, kernels_der, biases_der


if __name__ == "__main__":
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
        

    for i in range(2):
        start = time.time()
        dense_output = dense_forward(dense_input, dense_weights, dense_biases)
        dense_input_der, dense_weights_der, dense_biases_der = dense_backward(dense_output_der, dense_input, dense_weights)

        if i == 0:
            continue

        print(f"Dense Elapsed time: {time.time() - start} seconds")
        assert dense_output.shape == dense_output_der.shape
        assert dense_input.shape == dense_input_der.shape
        assert dense_weights.shape == dense_weights_der.shape
        assert dense_biases.shape == dense_biases_der.shape

    flatten_output = flatten_forward(flatten_input)
    flatten_input2 = flatten_backward(flatten_output, flatten_input.shape)
    assert flatten_output.shape == (flatten_input.shape[0], np.prod(flatten_input.shape[1:]))
    assert np.allclose(flatten_input, flatten_input2)















