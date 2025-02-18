"""Simple tools to manage the model. (open / load / save / ...)"""

import os

import pandas as pd


def load_model(model_path: str):
    """
    Load model from a file.
    Parameters
    ----------
    model_path : The file path.

    Returns
    -------
    The loaded model.
    """
    return pd.read_csv(model_path, index_col=[0, 1])


def save_model(model: pd.DataFrame, file_name: str) -> None:
    """
    Save the model as a json file.
    Parameters
    ----------
    model : The dataframe with the model to save.
    file_name : The destination file.

    Returns
    -------
    None
    """
    model.to_csv(file_name)
