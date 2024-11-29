import ops
import time
import layers
import numpy as np


class Model:
    
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = self.layers[0].forward(input)
        for i in range(1, len(self.layers)):
            output = self.layers[i].forward(output)

        return output

    def backward(self, output_der):
        input_der = self.layers[-1].backward(output_der)
        for i in range(len(self.layers)-2, -1, -1):
            input_der = self.layers[i].backward(input_der)

        return input_der

    def train_step(self, step_size):
        for i in range(len(self.layers)):
            self.layers[i].train_step(step_size)

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass


if __name__ == '__main__':

    model = Model()
    model.add(layers.ConvLayer(5, 1, 20, ops.relu, ops.relu_der))
    model.add(layers.MaxPoolLayer(2))
    model.add(layers.ConvLayer(5, 20, 40, ops.relu, ops.relu_der))
    model.add(layers.MaxPoolLayer(2))
    model.add(layers.FlattenLayer())
    model.add(layers.DenseLayer(40*4*4, 100, ops.relu, ops.relu_der))
    model.add(layers.DenseLayer(100, 10, ops.linear, ops.linear_der))

    input = np.random.normal(size=(32, 28, 28, 1))
    output_der = np.random.normal(size=(32, 10))

    _ = model.forward(input)
    _ = model.backward(output_der)

    start = time.time()
    output = model.forward(input)
    input_der = model.backward(output_der)
    model.train_step(0.001)
    print(f"Full batch pass with step: {time.time() - start:.3f}")



