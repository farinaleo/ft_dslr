import numpy as np
import pandas as pd


def sigmoid(z: pd.Series) -> pd.Series:
    """
    Compute the sigmoid function
    :param z: pd.Series, the input
    :return: pd.Series, the output
    """
    return z.apply(lambda x: 1 / (1 + np.exp(-x)))
