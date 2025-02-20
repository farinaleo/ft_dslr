"""Mandatory batch selector."""

import numpy as np
import pandas as pd


def mandatory_batch(
    X: pd.Series | np.ndarray, Y: pd.Series | np.ndarray, **args
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mandatory batch selector. The selector returns the entire dataset.
    Parameters
    ----------
     X : The feature.
    Y : The target.

    Returns
    -------
    The selected feature and target. (X, Y)
    """

    return X, Y
