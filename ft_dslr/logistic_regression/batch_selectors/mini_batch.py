"""The stochastic batch return a unique row (X and Y) picked randomly."""

import numpy as np
import pandas as pd

MINI_BATCH_SIZE = 32


def mini_batch(X: pd.DataFrame, Y: pd.DataFrame, **args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mini batch return a small batch of N elements (X and Y) picked randomly.
    Parameters
    ----------
     X : The feature.
    Y : The target.

    Returns
    -------
    The selected feature and target. (X, Y).
    """

    list_id = np.random.choice(X.index.tolist(), size=MINI_BATCH_SIZE, replace=False)

    list_id = [int(id) for id in list_id.tolist()]

    _X = X.loc[list_id]
    _Y = Y.loc[:, list_id]

    return _X, _Y
