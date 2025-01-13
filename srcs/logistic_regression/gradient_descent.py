import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.sigmoid import sigmoid


def gradient_descent(
    X: pd.Series, Y: pd.Series, learning_rate: float, epoch: int
) -> dict[str, float]:
    """
    Compute the gradient descent of the logistic regression
    :param X: pd.Series, the feature
    :param Y: pd.Series, the target
    :param learning_rate: float, the learning rate
    :param epoch: int, the number of iterations
    :return: dict, a dictionary containing the thetas
    """
    m = len(Y)
    thetas = {"theta0": 0, "theta1": 1}

    for _ in range(epoch):
        z = X.apply(lambda x: thetas["theta0"] + thetas["theta1"] * x)
        h = sigmoid(z)

        thetas["theta0"] -= learning_rate * (1 / m) * np.sum(h - Y)
        thetas["theta1"] -= learning_rate * (1 / m) * np.sum(np.transpose(h - Y) * X)

    return thetas
