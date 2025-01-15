"""Tools ot normalise a dataframe."""

import pandas as pd


def normalise_df(df: pd.Series) -> pd.Series:
    """
    Normalise a dataframe to obtain a μ = 0 and σ = 1 with the Z-score.
    Parameters
    ----------
    df : The dataframe to be normalised.

    Returns
    -------
    The normalised dataframe.
    """
    df_cp = df.copy(deep=True)

    _mean = df_cp.mean()
    _std = df_cp.std()
    df_cp = df_cp.apply(lambda x: (x - _mean) / _std)

    return df_cp


def denorm_thetas(thetas: dict, mean_x: float, std_x: float, mean_y: float, std_y: float) -> dict:
    """
    De-normalise thetas trained on a normalised dataframe.
    Parameters
    ----------
    thetas : Dictionary of thetas.
    mean_x : The mean of the original data to explain.
    std_x : The standard deviation of the original data to explain.
    mean_y : The mean of the original data to predict.
    std_y : The standard deviation of the original data to predict.

    Returns
    -------
    A dictionary with thetas de-normalised.
    """
    t1 = thetas["theta1"] * (std_y / std_x)
    t0 = mean_y + std_y * (thetas["theta0"] - (thetas["theta1"] * (mean_x / std_x)))

    return {"theta0": t0, "theta1": t1}
