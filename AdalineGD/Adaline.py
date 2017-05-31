import numpy as np

class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta # Learning rate
        self.n_iter = n_iter # Passes over training set
        self.w_ = [] # Weights after fitting
        self.errors_ = [] # Num of miss

    """
    X: [n_samples, n_features] Training vectors
    y: [n_samples] Target values
    """
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)

            # Calculate gradient based on whole training set for weights 1 to m
            # X.T.dot(errors) is a matrix-vector multiplication between feature matrix
            # and error vector.
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # We collect the cost values in a lis to check if the algo is converged
            # after training.
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    """
    Calculate net input
    """
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """
    Compute linear activation
    """
    def activation(self, X):
        return self.net_input(X)

    """
    Return class label after unit step
    """
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
