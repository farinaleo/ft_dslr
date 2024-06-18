import pandas as pd
import numpy as np
from tools.sigmoid import sigmoid
import matplotlib.pyplot as plt


def gradient_descent(X: pd.Series, Y: pd.Series, theta: list, learning_rate: float, epoch: int) -> list:
    """
    Compute the gradient descent of the logistic regression
    :param X: pd.Series, the feature
    :param Y: pd.Series, the target
    :param theta: list, the theta
    :param learning_rate: float, the learning rate
    :param epoch: int, the number of iterations
    :return: list, the updated thetas
    """
    m = len(Y)
    thetas = theta

    for _ in range(epoch):
        z = X.apply(lambda x: thetas[0] + thetas[1] * x)
        h = sigmoid(z)

        thetas[0] -= learning_rate * (1 / m) * np.sum(h - Y)
        thetas[1] -= learning_rate * (1 / m) * np.sum(np.transpose(h - Y) * X)

    return thetas
