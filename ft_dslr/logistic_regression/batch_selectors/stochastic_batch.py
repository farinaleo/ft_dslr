"""The stochastic batch return a unique row (X and Y) picked randomly."""

import numpy as np
import pandas as pd


def stochastic_batch(
    X: pd.DataFrame, Y: pd.DataFrame, **args
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    r_id = np.random.choice(X.index.tolist(), 1)

    _X = X.loc[[int(r_id[0])]]
    _Y = Y.loc[:, [int(r_id[0])]]

    return _X, _Y
