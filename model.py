import ops
import time
import pickle
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

    def save_weights(self, path='model_params.pickle'):
        params_dict = {}
        for i in range(len(self.layers)):
            layers_params = self.layers[i].to_dict()
            if layers_params is not None:
                params_dict[f'layer{i}'] = layers_params

        with open(path, 'wb') as f:
            pickle.dump(params_dict, f)

    def load_weights(self, path='model_params.pickle'):
        with open(path, 'rb') as f:
            params_dict = pickle.load(f)

        for i in range(len(self.layers)):
            layer_params = params_dict.get(f'layer{i}', None)
            if layer_params is not None:
                self.layers[i].from_dict(layer_params)


if __name__ == '__main__':
    def get_model():
        model = Model()
        model.add(layers.ConvLayer(5, 1, 20, ops.relu, ops.relu_der))
        model.add(layers.MaxPoolLayer(2))
        model.add(layers.ConvLayer(5, 20, 40, ops.relu, ops.relu_der))
        model.add(layers.MaxPoolLayer(2))
        model.add(layers.FlattenLayer())
        model.add(layers.DenseLayer(40*4*4, 100, ops.relu, ops.relu_der))
        model.add(layers.DenseLayer(100, 10, ops.linear, ops.linear_der))
        return model

    input = np.random.normal(size=(32, 28, 28, 1))
    output_der = np.random.normal(size=(32, 10))

    model = get_model()
    _ = model.forward(input)
    _ = model.backward(output_der)

    start = time.time()
    output = model.forward(input)
    input_der = model.backward(output_der)
    model.train_step(0.001)
    print(f"Full batch pass with step: {time.time() - start:.3f}")

    output = model.forward(input)
    model.save_weights('test.pickle')
    model1 = get_model()
    model1.load_weights('test.pickle')
    output1 = model1.forward(input)
    assert np.allclose(output, output1)



