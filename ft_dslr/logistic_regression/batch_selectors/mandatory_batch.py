"""Mandatory batch selector."""

import pandas as pd


def mandatory_batch(X: pd.Series, Y: pd.Series, **args) -> tuple[pd.Series, pd.Series]:
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
