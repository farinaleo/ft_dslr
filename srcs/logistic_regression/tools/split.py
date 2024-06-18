import pandas as pd


def split_date(data: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Split the data into training and testing sets
    :param data: the data to split
    :param test_size: the size of the testing set
    :param random_state: the seed for the random number generator
    :return: the training and testing sets X_train, X_test, y_train, y_test
    """

    data_sample = data.sample(frac=1, random_state=random_state)

    X_train = data_sample.iloc[:int(len(data_sample) * (1 - test_size))]
    X_test = data_sample.iloc[int(len(data_sample) * (1 - test_size)):]

    y_train = X_train['Hogwarts House']
    y_test = X_test['Hogwarts House']

    X_train = X_train.drop(columns=['Hogwarts House'])
    X_test = X_test.drop(columns=['Hogwarts House'])

    return X_train, X_test, y_train, y_test
