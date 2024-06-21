import pandas as pd


def normalise_df(df: pd.Series) -> pd.Series:
    """
    Normalise a dataframe to obtain a μ = 0 and σ = 1 with the Z-score.
    :param df: The dataframe to normalise.
    :return: The normalised dataframe.
    """

    df_cp = df.copy(deep=True)

    _mean = df_cp.mean()
    _std = df_cp.std()
    df_cp = df_cp.apply(lambda x: (x - _mean) / _std)

    return df_cp


def denorm_thetas(thetas: dict, mean_x: float, std_x: float, mean_y: float, std_y: float) -> dict:
    """
    Denormalise thetas trained on a normalised dataframe.
    :param std_y: the standard deviation of the data to predict.
    :param mean_y: the mean of the data to predict.
    :param std_x: the standard deviation of the explain data.
    :param mean_x: the standard deviation of the explain data.
    :param thetas: normalised thetas.
    :return: Denormalised thetas.
    """
    t1 = thetas['theta1'] * (std_y / std_x)
    t0 = mean_y + std_y * (thetas['theta0'] - (thetas['theta1'] * (mean_x / std_x)))

    return {'theta0': t0, 'theta1': t1}
