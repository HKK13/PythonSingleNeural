from numpy.random import seed, permutation
from numpy.core import zeros, dot, where


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta # Learning rate.
        self.n_iter = n_iter # Passes over dataset.
        self.shuffle = shuffle # Shuffles training data every epoch
        self.w_ = [] # Weights after fitting
        self.errors_ = [] # Num off miss

        if random_state:
            seed(random_state) # Random state for shuffling and weight init

    """
    Fits training data.
    """
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    """
    Fit without reinitializing weights
    """
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = permutation(len(y)) # Generate a random sequence in range 0, 100
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = zeros(1 + m)
        self.w_initialized =True

    """
    Apply Adaline learning rule
    """
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        return dot(X, self.w_[1:]) + self.w_[0]

    """
    Compute linear activation
    """
    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return where(self.activation(X) >= 0.0, 1, -1)
