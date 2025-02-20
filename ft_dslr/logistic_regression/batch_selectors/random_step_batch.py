import numpy as np
import pandas as pd


def random_step_batch(
    X: pd.Series | np.ndarray, Y: pd.Series | np.ndarray, **args
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random batch selector. The selector returns a random selection of 20% of elements
    with a random step between 1 and 10.
    Parameters
    ----------
    X : The feature.
    Y : The target.

    Returns
    -------
    The selected feature and target. (X, Y)
    """

    indices = np.arange(0, len(X), np.random.randint(1, 11))
    np.random.shuffle(indices)
    selected_indices = indices[:int(len(X) * 0.2)]

    _X = X.iloc[selected_indices]
    _Y = Y.iloc[selected_indices]

    return _X, _Y
