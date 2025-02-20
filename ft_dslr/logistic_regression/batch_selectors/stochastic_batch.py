"""The stochastic batch return a unique row (X and Y) picked randomly."""

import random

import numpy as np
import pandas as pd


def stochastic_batch(
    X: pd.Series | np.ndarray, Y: pd.Series | np.ndarray, **args
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic batch return a unique row (X and Y) picked randomly.
    Parameters
    ----------
     X : The feature.
    Y : The target.

    Returns
    -------
    The selected feature and target. (X, Y).
    """

    r_id = random.randint(0, len(X) - 1)

    _X = X.iloc[[r_id]]
    _Y = Y.iloc[[r_id]]

    return _X, _Y
