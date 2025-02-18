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
    pbar=None,
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
    pbar : A progress bar.
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

        if pbar is not None:
            pbar.update(1)

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

    _X = X.to_numpy()
    _Y = Y.to_numpy()

    z = theta0 + theta1 * _X
    h = sigmoid(z)
    error = h - _Y

    theta0 -= learning_rate * np.mean(error)
    theta1 -= learning_rate * np.dot(error, X) / m

    return theta0, theta1
