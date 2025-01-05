import ops
import utils
import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = utils.load_mnist()
model = utils.get_model()
model.load_weights('model_params.pickle')


def get_integrated_gradients(X, N=20):
    integrated_gradients = np.zeros_like(X)
    for j in range(1, N+1):
        output = model.forward(X*j/N)
        output = ops.softmax(output)
        pred = np.argmax(output, axis=1)
        output_grad = np.zeros_like(output)
        for i in range(X.shape[0]):
            output_grad[i, :] *= -output[i, pred[i]]
            output_grad[i, pred[i]] = output[i, pred[i]] * (1 - output[i, pred[i]])
        
        input_grad = model.backward(output_grad)
        integrated_gradients += input_grad
    
    return integrated_gradients * X/N, pred

integrated_gradients, pred = get_integrated_gradients(X_test[0:6])

fig, axs = plt.subplots(2, 3)

for k in range(pred.shape[0]):
    i, j = k//3, k%3
    axs[i, j].imshow(X_test[k, ..., 0], cmap='gray')
    axs[i, j].imshow(integrated_gradients[k], cmap='viridis', interpolation='none', alpha=0.6*(integrated_gradients[k] != 0)[..., 0])
    axs[i, j].set_title(f"True: {y_test[k]}, Pred: {pred[k]}")
    axs[i, j].axis('off')

plt.savefig('images/integrated_gradients.png', bbox_inches='tight', dpi=300)