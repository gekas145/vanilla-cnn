import ops
import numpy as np


class Layer:

    def __init__(self, activation, activation_grad):
        self.activation = activation
        self.activation_grad = activation_grad
        self.raw_output = None
        self.input = None

    def forward(self, input):
        pass

    def backward(self, output_grad):
        pass

    def train_step(self, step_size):
        pass

    def to_dict(self):
        return None

    def from_dict(self, params_dict):
        pass


class ConvLayer(Layer):

    def __init__(self, kernel_size, input_channels, output_channels, activation, activation_grad):

        super().__init__(activation, activation_grad)

        self.kernels = np.random.uniform(low=-0.1, high=0.1, size=(output_channels, kernel_size, kernel_size, input_channels))
        self.kernels_grad = np.zeros_like(self.kernels, dtype=np.float64)

        self.biases = np.random.uniform(low=-0.1, high=0.1, size=output_channels)
        self.biases_grad = np.zeros_like(self.biases, dtype=np.float64)

    def forward(self, input):
        output = ops.cnn_forward(input, self.kernels, self.biases)

        self.input = input
        self.raw_output = output

        return self.activation(output)
    
    def backward(self, output_grad):
        input_grad, kernels_grad, biases_grad = ops.cnn_backward(output_grad * self.activation_grad(self.raw_output), self.input, self.kernels)
        self.kernels_grad = kernels_grad
        self.biases_grad = biases_grad

        return input_grad
    
    def train_step(self, step_size):
        self.kernels -= self.kernels_grad * step_size
        self.biases -= self.biases_grad * step_size

        self.kernels_grad *= 0.0
        self.biases_grad *= 0.0

    def to_dict(self):
        return {'kernels': self.kernels.tolist(), 'biases': self.biases.tolist()}
    
    def from_dict(self, params_dict):
        self.kernels = np.array(params_dict['kernels'], dtype=np.float64)
        self.biases = np.array(params_dict['biases'], dtype=np.float64)


class DenseLayer(Layer):

    def __init__(self, input_size, output_size, activation, activation_grad):

        super().__init__(activation, activation_grad)

        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size))
        self.weights_grad = np.zeros_like(self.weights, dtype=np.float64)

        self.biases = np.random.uniform(low=-0.1, high=0.1, size=output_size)
        self.biases_grad = np.zeros_like(self.biases, dtype=np.float64)

    def forward(self, input):
        output = ops.dense_forward(input, self.weights, self.biases)

        self.input = input
        self.raw_output = output

        return self.activation(output)
    
    def backward(self, output_grad):
        input_grad, weights_grad, biases_grad = ops.dense_backward(output_grad * self.activation_grad(self.raw_output), self.input, self.weights)
        self.weights_grad = weights_grad
        self.biases_grad = biases_grad

        return input_grad
    
    def train_step(self, step_size):
        self.weights -= self.weights_grad * step_size
        self.biases -= self.biases_grad * step_size

        self.weights_grad *= 0.0
        self.biases_grad *= 0.0

    def to_dict(self):
        return {'weights': self.weights.tolist(), 'biases': self.biases.tolist()}
    
    def from_dict(self, params_dict):
        self.weights = np.array(params_dict['weights'], dtype=np.float64)
        self.biases = np.array(params_dict['biases'], dtype=np.float64)
        

class MaxPoolLayer(Layer):

    def __init__(self, window_size):

        super().__init__(None, None)

        self.window_size = window_size
        self.indexes = None
        self.input_shape = None

    def forward(self, input):
        output, indexes = ops.maxpool_forward(input, self.window_size)
        self.indexes = indexes
        self.input_shape = input.shape
        
        return output

    def backward(self, output_grad):
        return ops.maxpool_backward(output_grad, self.input_shape, self.indexes)
    

class FlattenLayer(Layer):

    def __init__(self):

        super().__init__(None, None)

        self.input_shape = None

    def forward(self, input):
        output = ops.flatten_forward(input)
        self.input_shape = input.shape
        
        return output

    def backward(self, output_grad):
        return ops.flatten_backward(output_grad, self.input_shape)
    


if __name__ == '__main__':
    kernel_size = 5
    input_channels = 1 
    output_channels = 20 
    activation = ops.sigmoid
    activation_grad = ops.sigmoid_grad

    batch_size = 32
    conv_input = np.random.normal(size=(batch_size, 28, 28, input_channels))

    conv = ConvLayer(kernel_size, input_channels, output_channels, activation, activation_grad)

    conv_output = conv.forward(conv_input)
    conv_input_grad = conv.backward(conv_output)
    assert conv_input.shape == conv_input_grad.shape

    maxpool = MaxPoolLayer(2)
    maxpool_output = maxpool.forward(conv_output)
    maxpool_input_grad = maxpool.backward(maxpool_output)
    assert conv_output.shape == maxpool_input_grad.shape
    
    flatten = FlattenLayer()
    flatten_output = flatten.forward(maxpool_output)
    flatten_input_grad = flatten.backward(flatten_output)
    assert np.allclose(maxpool_output, flatten_input_grad)

    dense = DenseLayer(flatten_output.shape[1], 100, ops.linear, ops.linear_grad)
    dense_output = dense.forward(flatten_output)
    dense_input_grad = dense.backward(dense_output)
    assert dense_output.shape == (batch_size, 100)
    assert flatten_output.shape == dense_input_grad.shape

