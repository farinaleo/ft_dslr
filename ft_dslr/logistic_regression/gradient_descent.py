"""The implementation of the gradient descent."""

import numpy as np
import pandas as pd

from ft_dslr.logistic_regression.tools import sigmoid


def gradient_descent(
    X: pd.DataFrame,
    Y: pd.Series,
    learning_rate: float,
    epoch: int,
) -> pd.DataFrame:
    """
    Compute the gradient descent of the logistic regression.
    Parameters
    ----------
    X : The features.
    Y : The target
    learning_rate : The learning rate.
    epoch : The number of epoch.

    Returns
    -------
    A pandas DataFrame containing the gradient descent thetas.
    """

    m = len(Y)

    thetas = pd.DataFrame(index=[0, 1], columns=X.columns)
    thetas.iloc[0] = 1
    thetas.iloc[1] = 2

    for _ in range(epoch):
        thetas = X.apply(
            lambda x: get_gradients(
                x, Y, thetas[x.name].loc[0], thetas[x.name].loc[1], m, learning_rate
            ),
            axis=0,
        )

    return thetas


def get_gradients(
    X: pd.Series, Y: pd.Series, theta0: float, theta1: float, m: int, learning_rate: float
) -> tuple[float, float]:
    """
    Compute the new gradient for a specific feature.
    Parameters
    ----------
    X : The feature.
    Y : The target.
    theta0 : First parameter to optimize.
    theta1 : Second parameter to optimize.
    m : Number of elements.
    learning_rate : The learning rate.

    Returns
    -------
    The thetas optimized.
    """

    z = X.apply(lambda x: theta0 + theta1 * x)
    h = sigmoid(z)

    theta0 -= learning_rate * (1 / m) * np.sum(h - Y)
    theta1 -= learning_rate * (1 / m) * np.sum(np.transpose(h - Y) * X)

    return theta0, theta1
