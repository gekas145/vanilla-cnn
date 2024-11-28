import ops
import numpy as np


class Layer:

    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der
        self.raw_output = None
        self.input = None

    def forward(self, input):
        pass

    def backward(self, output_der):
        pass

    def train_step(self, step_size):
        pass

    def to_dict(self):
        return None

    def from_dict(self, params_dict):
        pass


class ConvLayer(Layer):

    def __init__(self, kernel_size, input_channels, output_channels, activation, activation_der):

        super().__init__(activation, activation_der)

        self.kernels = np.random.normal(loc=0., scale=1.0, size=(output_channels, kernel_size, kernel_size, input_channels))
        self.kernels_der = np.zeros_like(self.kernels, dtype=np.float32)

        self.biases = np.random.normal(loc=0., scale=1.0, size=output_channels)
        self.biases_der = np.zeros_like(self.biases, dtype=np.float32)

    def forward(self, input):
        output = ops.cnn_forward(input, self.kernels, self.biases)

        self.input = input
        self.raw_output = output

        return self.activation(output)
    
    def backward(self, output_der):
        input_der, kernels_der, biases_der = ops.cnn_backward(output_der * self.activation_der(self.raw_output), self.input, self.kernels)
        self.kernels_der = kernels_der
        self.biases_der = biases_der

        return input_der
    
    def train_step(self, step_size):
        self.kernels += self.kernels_der * step_size
        self.biases += self.biases_der * step_size

        self.kernels_der *= 0.0
        self.biases_der *= 0.0

    def to_dict(self):
        return {'kernels': self.kernels.tolist(), 'biases': self.biases.tolist()}
    
    def from_dict(self, params_dict):
        self.kernels = params_dict['kernels']
        self.biases = params_dict['biases']
        

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

    def backward(self, output_der):
        return ops.maxpool_backward(output_der, self.input_shape, self.indexes)
    


if __name__ == '__main__':
    kernel_size = 5
    input_channels = 1 
    output_channels = 20 
    activation = ops.sigmoid
    activation_der = ops.sigmoid_der

    batch_size = 32
    conv_input = np.random.normal(size=(batch_size, 28, 28, input_channels))

    conv = ConvLayer(kernel_size, input_channels, output_channels, activation, activation_der)

    conv_output = conv.forward(conv_input)
    conv_input_der = conv.backward(conv_output)
    assert conv_input.shape == conv_input_der.shape

    maxpool = MaxPoolLayer(2)
    maxpool_output = maxpool.forward(conv_output)
    maxpool_input_der = maxpool.backward(maxpool_output)
    assert conv_output.shape == maxpool_input_der.shape
    print(maxpool_output.shape)

