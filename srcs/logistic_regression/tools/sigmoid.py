import numpy as np
import pandas as pd


def sigmoid(z: pd.Series) -> pd.Series:
    """
    Compute the sigmoid function
    :param z: float, the input
    :return: float, the sigmoid of z
    """
    return z.apply(lambda x: 1 / (1 + np.exp(-x)))
