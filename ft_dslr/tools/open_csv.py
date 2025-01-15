"""Read csv file."""

import pandas as pd


def open_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Open csv file.
    Parameters
    ----------
    path : Path of the csv file.
    kwargs : Arguments passed to pd.read_csv.

    Returns
    -------
    The csv as a dataframe.
    """
    df = pd.read_csv(path, **kwargs)
    return df
