"""The stochastic batch return a unique row (X and Y) picked randomly."""

import numpy as np
import pandas as pd

MINI_BATCH_SIZE = 32


def mini_batch(X: pd.Series, Y: pd.Series, **args) -> tuple[pd.Series, pd.Series]:
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

    return X.iloc[list_id], Y.iloc[list_id]
