import pandas as pd


def split_date(data: pd.DataFrame, test_size: float, random_state: int):
    """
    Split the data into training and testing sets
    :param data: the data to split
    :param test_size: the size of the testing set in percentage
    :param random_state: the seed for the random number generator
    :return: the training and testing sets X_train, X_test, y_train, y_test
    """
    frac = test_size / 100.0

    X_test = data.sample(frac=frac, random_state=random_state)
    y_test = X_test['Hogwarts House']

    X_index = X_test.index

    X_train = data.drop(X_index)
    y_train = X_train['Hogwarts House']

    X_train = X_train.drop(columns=['Hogwarts House'])
    X_test = X_test.drop(columns=['Hogwarts House'])

    return X_train, X_test, y_train, y_test
