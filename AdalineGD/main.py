import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from Adaline import AdalineGD


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


def compare_learning_rate(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # With large learning rate
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log(Sum-Squared-Error)")
    ax[0].set_title("AdalineGD - Learning rate 0.01")

    # With small learning rate
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Sum-Squared-Error")
    ax[1].set_title("AdalineGD - Learning rate 0.0001")


def adaline_With_standardization(X, y):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Sum Squared Error")


def main():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                     header=None)
    df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values



    plt.show()


if __name__ == '__main__':
    main()