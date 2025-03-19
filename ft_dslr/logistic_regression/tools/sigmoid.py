"""Sigmoid formula as function."""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def sigmoid(z: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Compute sigmoid function.
    Parameters
    ----------
    z : A series of values.

    Returns
    -------
    A series of results.
    """

    z = np.asarray(z)
    return 1 / (1 + np.exp(-z))


def simple_sigmoid(x: float, a: float, b: float) -> float:
    """
    Compute simple sigmoid function.
    Parameters
    ----------
    x : The variable.
    a : The coefficient.
    b : The intercept.

    Returns
    -------
        The computed value.
    """
    return float(1 / (1 + np.exp(-(b + a * x))))
