import ops
import utils
import numpy as np
from tqdm import tqdm

batch_size = 32
epochs = 10
step_size = 0.001

X_train, X_test, y_train, y_test = utils.load_mnist()
model = utils.get_model()
train_data_idxs = np.array(range(X_train.shape[0]))
best_score = -np.inf

for epoch in range(1, epochs+1):
    np.random.shuffle(train_data_idxs)

    for batch in tqdm(range(0, X_train.shape[0], batch_size), ncols=80):
        X_batch = X_train[train_data_idxs[batch:batch + batch_size], ...]
        y_batch = y_train[train_data_idxs[batch:batch + batch_size]]
        y_batch_onehot = np.zeros((y_batch.shape[0], 10), dtype=np.float64)
        y_batch_onehot[np.arange(y_batch.shape[0]), y_batch] = 1.0

        output = model.forward(X_batch)
        output = ops.softmax(output)
        output_grad = (y_batch_onehot - output) / float(X_batch.shape[0]) # multiclass cross-entropy derivative by logits
        model.backward(output_grad)
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

