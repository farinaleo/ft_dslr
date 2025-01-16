"""The implementation of stochastic gradient descent."""

import numpy as np
import pandas as pd

from ft_dslr.logistic_regression.tools import sigmoid


def stochastic_gradient_descent(
    X: pd.Series, Y: pd.Series, learning_rate: float, epoch: int
) -> dict[str, float]:
    """
    Compute the stochastic gradient descent of the logistic regression.
    Parameters
    ----------
    X : The feature.
    Y : The target.
    learning_rate : The learning rate.
    epoch : The number of epoch.

    Returns
    -------
    A dictionary containing the stochastic gradient descent thetas.
    Parameters
    ----------
    X :
    Y :
    learning_rate :
    epoch :

    Returns
    -------

    """
    thetas = {"theta0": 0, "theta1": 1}

    return thetas
