import ops
import layers
import zipfile
import numpy as np
from model import Model

def get_model():
    model = Model()
    model.add(layers.ConvLayer(5, 1, 20, ops.relu, ops.relu_grad))
    model.add(layers.MaxPoolLayer(2))
    model.add(layers.ConvLayer(5, 20, 40, ops.relu, ops.relu_grad))
    model.add(layers.MaxPoolLayer(2))
    model.add(layers.FlattenLayer())
    model.add(layers.DenseLayer(40*4*4, 100, ops.relu, ops.relu_grad))
    model.add(layers.DenseLayer(100, 10, ops.linear, ops.linear_grad))
    return model


def load_mnist():
    def prepare_images(buffer):
        prepared = np.frombuffer(buffer, offset=16, dtype=np.uint8).astype(np.float64)
        return prepared.reshape(len(buffer) // image_size ** 2, image_size, image_size, 1) / 255

    image_size = 28

    file_names = ['t10k-images.idx3-ubyte',
                  't10k-labels.idx1-ubyte',
                  'train-images.idx3-ubyte',
                  'train-labels.idx1-ubyte']
    data = []
    with zipfile.ZipFile('MNIST.zip', 'r') as zf:
        for file in file_names:
            with zf.open(file, 'r') as f:
                data.append(f.read())
                f.close()
        zf.close()
    del f
    del zf

    X_train = prepare_images(data[2])
    X_test = prepare_images(data[0])

    y_train = np.frombuffer(data[3], offset=8, dtype=np.uint8).astype(np.int64)
    y_test = np.frombuffer(data[1], offset=8, dtype=np.uint8).astype(np.int64)

    return X_train, X_test, y_train, y_test