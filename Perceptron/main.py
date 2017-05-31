import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Perceptron import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
        # Create a color map from the list of colors
        markers = ("s", "x", "o", "^", "v")
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Determine min and max values for the two features
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Use features to create a pair of grid arrays xx1 and xx2
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        # Create a matrix that has the same num of cols as the training subset
        # so that we can use predict method.
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        # ravel returns a contiguous flattened array. Read in order
        Z = Z.reshape(xx1.shape)

        # Draw a contour that maps the diff decision regions to diff colors
        # for each predicted class in the grid array.
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)


def draw_errors(X, y):
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Num Of Misclass")
    plt.show()


def main():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                     header=None)
    df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    """
    Moon above
    """

    plot_decision_regions(X, y, classifier=ppn)

    """
    Sun below
    """
    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")
    plt.show()



if __name__ == '__main__':
    main()