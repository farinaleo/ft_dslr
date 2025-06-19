"""Split a dataframe to have a train and a test part."""

import pandas as pd


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple:
    """
    Split a dataframe to have a train and a test part.
    Parameters
    ----------
    data : The source dataframe.
    test_size : The size of the test split. (as a ratio)
    random_state : The seed of the ramdom number generator.

    Returns
    -------
    A tuple of the train and test dataframes.
    """

    data_sample = data.sample(frac=1, random_state=random_state)

    X_train = data_sample.iloc[: int(len(data_sample) * (1 - test_size))]
    X_test = data_sample.iloc[int(len(data_sample) * (1 - test_size)):]

    y_train = X_train["Hogwarts House"]
    y_test = X_test["Hogwarts House"]

    X_train = X_train.drop(columns=["Hogwarts House"])
    X_test = X_test.drop(columns=["Hogwarts House"])

    return X_train, X_test, y_train, y_test
