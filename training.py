import ops
import layers
import zipfile
import numpy as np
from tqdm import tqdm
from model import Model

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

batch_size = 32
epochs = 10
step_size = 0.0001

X_train, X_test, y_train, y_test = load_mnist()
model = get_model()
train_data_idxs = np.array(range(X_train.shape[0]))
best_score = -np.inf

for epoch in range(1, epochs+1):
    np.shuffle(train_data_idxs)

    for batch in tqdm(range(0, X_train.shape[0], batch_size), ncols=80):
        X_batch = X_train[train_data_idxs[batch:batch + batch_size], ...]
        y_batch = y_train[train_data_idxs[batch:batch + batch_size]]
        y_pred = np.zeros((y_batch.shape[0], 10), dtype=np.float64)
        y_pred[np.arange(y_batch.shape[0]), y_batch] = 1.0

        output = model.forward(X_batch)
        output = ops.softmax(output)
        output_der = (output - y_pred) / float(batch_size) # multiclass cross-entropy derivative by logits
        model.backward(output_der)
        model.train_step(step_size)

    train_correct, test_correct = 0, 0
    for batch in tqdm(range(0, X_test.shape[0], batch_size), ncols=80):
        X_batch_train = X_train[train_data_idxs[batch:batch + batch_size], ...]
        y_batch_train = y_train[train_data_idxs[batch:batch + batch_size]]
        output_train = np.argmax(model.forward(X_batch_train), axis=-1)
        train_correct += np.sum(output_train == y_batch_train)


        X_batch_test = X_test[batch:batch + batch_size, ...]
        y_batch_test = y_test[batch:batch + batch_size]
        output_test = np.argmax(model.forward(X_batch_test), axis=-1)
        test_correct += np.sum(output_test == y_batch_test)

    train_correct = train_correct/X_test.shape[0]
    test_correct = test_correct/X_test.shape[0]
    print(f"[Epoch {epoch}] train accuracy: {train_correct:.3f}, test accuracy: {test_correct:.3f}")
    if best_score < test_correct:
        model.save_weights()
        best_score = test_correct

