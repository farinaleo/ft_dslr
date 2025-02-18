"""Simple functions to manage data."""

import configparser

import pandas as pd

from .format_csv import config_drop_columns


def prepare_data(df_data: pd.DataFrame, config: configparser.ConfigParser):
    """
    Prepare the data for the prediction.
    Parameters
    ----------
    df_data : The date to prepare.
    config : The configuration to apply.

    Returns
    -------
    The prepared data.
    """
    df_data = config_drop_columns(df_data, config, verbose=False)
    X = df_data["Hogwarts House"].copy(deep=True)
    df_data.drop("Hogwarts House", axis=1, inplace=True)
    return df_data, X
