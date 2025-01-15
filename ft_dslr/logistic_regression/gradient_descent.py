"""The implementation of the gradient descent."""

import numpy as np
import pandas as pd

from ft_dslr.logistic_regression.tools import sigmoid


def gradient_descent(
    X: pd.Series, Y: pd.Series, learning_rate: float, epoch: int
) -> dict[str, float]:
    """
    Compute the gradient descent of the logistic regression.
    Parameters
    ----------
    X : The feature.
    Y : The target.
    learning_rate : The learning rate.
    epoch : The number of epoch.

    Returns
    -------
    A dictionary containing the gradient descent thetas.
    """
    m = len(Y)
    thetas = {"theta0": 0, "theta1": 1}

    for _ in range(epoch):
        z = X.apply(lambda x: thetas["theta0"] + thetas["theta1"] * x)
        h = sigmoid(z)

        thetas["theta0"] -= learning_rate * (1 / m) * np.sum(h - Y)
        thetas["theta1"] -= learning_rate * (1 / m) * np.sum(np.transpose(h - Y) * X)

    return thetas
