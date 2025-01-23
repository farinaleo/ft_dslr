"""The stochastic batch return a unique row (X and Y) picked randomly."""

import random

import pandas as pd


def stochastic_batch(X: pd.Series, Y: pd.Series, **args) -> tuple[pd.Series, pd.Series]:
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

    return X.iloc[[r_id]], Y.iloc[[r_id]]
