import numpy as np
import pandas as pd


def sigmoid(z: pd.Series) -> pd.Series:
    """
    Compute the sigmoid function
    :param z: pd.Series, a series of values
    :return: pd.Series, the result of the sigmoid function
    """
    return z.apply(lambda x: 1 / (1 + np.exp(-x)))
