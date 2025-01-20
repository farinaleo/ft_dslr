"""The implementation of the gradient descent."""

from typing import Callable

import numpy as np
import pandas as pd

from ft_dslr.logistic_regression.batch_selectors import mandatory_batch
from ft_dslr.logistic_regression.tools import sigmoid


def gradient_descent(
    X: pd.DataFrame,
    Y: pd.Series,
    learning_rate: float,
    epoch: int,
    batch_selector: Callable[[pd.DataFrame, pd.Series], tuple] = mandatory_batch,
) -> pd.DataFrame:
    """
    Compute the gradient descent of the logistic regression.
    Parameters
    ----------
    X : The features.
    Y : The target
    learning_rate : The learning rate.
    epoch : The number of epoch.
    batch_selector : A function that select a batch.

    Returns
    -------
    A pandas DataFrame containing the gradient descent thetas.
    """

    thetas = pd.DataFrame(index=[0, 1], columns=X.columns)
    thetas.iloc[0] = 0
    thetas.iloc[1] = 1

    for _ in range(epoch):
        _X, _Y = batch_selector(X, Y)
        thetas = _X.apply(
            lambda x: get_gradients(
                x, _Y, thetas[x.name].loc[0], thetas[x.name].loc[1], len(_Y), learning_rate
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
