"""Sigmoid formula as function."""

import numpy as np
import pandas as pd


def sigmoid(z: pd.Series) -> pd.Series:
    """
    Compute sigmoid function.
    Parameters
    ----------
    z : A series of values.

    Returns
    -------
    A series of results.
    """
    return z.apply(lambda x: 1 / (1 + np.exp(-x)))


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
