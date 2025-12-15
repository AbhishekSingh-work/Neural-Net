import numpy as np

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def prime(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

class CategoricalCrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def prime(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) / y_true.shape[0]
