"""Mandatory batch selector."""

import pandas as pd


def mandatory_batch(
    X: pd.DataFrame, Y: pd.DataFrame, **args
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
