"""The stochastic batch return a unique row (X and Y) picked randomly."""

import numpy as np
import pandas as pd

MINI_BATCH_SIZE = 32


def mini_batch(
    X: pd.Series | np.ndarray, Y: pd.Series | np.ndarray, **args
) -> tuple[np.ndarray, np.ndarray]:
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

    list_id = np.random.choice(len(Y), size=MINI_BATCH_SIZE, replace=False)

    _X = X.iloc[list_id]
    _Y = Y.iloc[list_id]
    _X = _X if isinstance(_X, np.ndarray) else _X.to_numpy()
    _Y = _Y if isinstance(_Y, np.ndarray) else _Y.to_numpy()
    return _X, _Y
