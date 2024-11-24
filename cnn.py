import time
import numba
import numpy as np
from numba import jit


B = 32
I = 28
C1 = 3
input = np.random.normal(loc=0., scale=1.0, size=(B, I, I, C1))

K = 3
C2 = 32
kernels = np.random.normal(loc=0., scale=1.0, size=(C2, K, K, C1))
biases = np.random.normal(loc=0., scale=1.0, size=C2)
output_der = np.random.normal(loc=0.0, scale=1.0, size=(B, I-K+1, I-K+1, C2))

def activation(x):
        return 1/(1 + np.exp(-x))

def activation_der(x):
        return 1/(1 + np.exp(-x))

@jit(nopython=True)
def forward(input, kernels, biases):
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
def backward(output_der, input, kernels):
    B = input.shape[0]
    I = input.shape[1]
    C = kernels.shape[0]
    K = kernels.shape[1]

    kernels_der = np.zeros_like(kernels, dtype=numba.float32)
    input_der = np.zeros_like(input, dtype=numba.float32)
    biases_der = output_der.reshape(B, -1).sum(axis=1)

    for b in range(B):
        for c in range(C):
            for s1 in range(0, I-K+1):
                for s2 in range(0, I-K+1):
                     kernels_der[c, ...] += output_der[b, s1, s2, c] * input[b, s1:s1+K, s2:s2+K, :]
                     input_der[b, s1:s1+K, s2:s2+K, :] = output_der[b, s1, s2, c] * kernels[c, ...]

    B = float(B)
    kernels_der = kernels_der/B
    biases_der = biases_der/B
    return input_der, kernels_der, biases_der


for i in range(10):
    start = time.time()
    # forward(input, kernels, biases)
    backward(output_der, input, kernels)
    print(f"Elapsed time: {time.time() - start} seconds")















