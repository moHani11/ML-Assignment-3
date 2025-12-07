import matplotlib.pyplot as plt
import numpy as np
from GaussianPredictor import GaussianDigitClassifier


def plot_two_features(X, y, feat1, feat2):

    f1 = X[:, feat1]
    f2 = X[:, feat2]

    classes = np.unique(y)

    plt.figure(figsize=(8, 6), dpi=200)

    alpha = 1
    linewidths = 1
    for c in classes:
        idx = (y == c)
        plt.scatter(
            f1[idx], 
            f2[idx],
            alpha=alpha,
            s=30,
            label=f"Class {c}",
            linewidths=linewidths

        )
        alpha-=0.1
        linewidths +=1

    plt.xlabel(f"Feature {feat1}")
    plt.ylabel(f"Feature {feat2}")
    plt.title(f"Feature {feat1} vs Feature {feat2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gaussian_accuracies(
    gaussian_classifier : GaussianDigitClassifier,
    X_train, y_train,
    X_test, y_test,
    lambdas=None,
):
    """
    Plots:
    1. Training accuracy vs Test accuracy
    2. Training accuracy vs lambda regularization values

    Parameters:
        gaussian_classifier: function that returns a fitted model
                        model = gaussian_classifier(X_train, y_train, reg_lambda)
                        model.predict(X) -> predicted labels
        X_train, y_train: training data
        X_test, y_test: test data
        lambdas: list/array of lambda values (default = logspace)
        dpi: plot resolution
    """

    if lambdas is None:
        lambdas = np.logspace(-5, 3, 10)

    train_accuracies = []
    test_accuracies = []

    for lam in lambdas:
        print(f"At lambda = {lam}\tIteration: {i}")
        gaussian_classifier.regularizeCovariance(lamda=lam, verbose=False)

        y_pred_train = gaussian_classifier.predict(X_train)
        y_pred_test = gaussian_classifier.predict(X_test)

        acc_train = (y_pred_train == y_train).mean()
        acc_test = (y_pred_test == y_test).mean()

        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)

    plt.figure(figsize=(10, 6))

    plt.plot(lambdas, train_accuracies, marker='o', label="Training Accuracy")
    plt.plot(lambdas, test_accuracies, marker='s', label="Test Accuracy")

    plt.xscale('log')

    plt.xlabel("Regularization Lambda")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Regularization (Gaussian Model)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
