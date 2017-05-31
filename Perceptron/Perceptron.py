import numpy as np


class Perceptron(object):
    eta = 0.0
    n_iter = 0
    w_ = [] # Weights after fitting.
    errors_ = [] # Number of misclassifications in every epoch

    """
    eta: Learning rate 0.0 < eta < 1.0
    n_iter: Passes over the raining dataset
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    """
    Fit training data
    X: [n_samples, n_features] Training vectors.
    y: [n_samples] Target values.
    """
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


    """
    Calculate net input
    """
    def net_input(self, X):
        # wT * x
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """
    Class label after unit step
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
