"""Read csv file."""

import pandas as pd


def open_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Open properly the given csv file.
    :param path: Path of the csv file.
    :param kwargs: Arguments for the dp.read_csv function.
    :return: The csv file as a pd.DataFrame.
    """
    df = pd.read_csv(path, **kwargs)
    return df
