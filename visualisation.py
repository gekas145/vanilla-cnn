import ops
import utils
import layers
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from model import Model

X_train, X_test, y_train, y_test = utils.load_mnist()
model = utils.get_model()
model.load_weights('model_params.pickle')

n = 6
output = model.forward(X_test[0:n])
output = ops.softmax(output)
pred = np.argmax(output, axis=1)
y_pred = np.zeros((n, 10), dtype=np.float64)
y_pred[np.arange(n), y_test[0:n]] = 1.0
output_der = (y_pred - output) / float(n)
input_der = model.backward(output_der)

fig, axs = plt.subplots(2, 3)

for k in range(n):
    i, j = k//3, k%3
    axs[i, j].imshow(X_test[k, ..., 0], cmap='gray', interpolation='none')
    axs[i, j].imshow(input_der[k, ..., 0], cmap='viridis', alpha=0.5*(input_der[k, ..., 0] > 0))
    axs[i, j].set_title(f"True: {y_test[k]}, Pred: {pred[k]}")
    axs[i, j].axis('off')

plt.savefig('images/input_grad.png', bbox_inches='tight', dpi=300)