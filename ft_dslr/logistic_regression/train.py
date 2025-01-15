"""Functions to train the logistic regression model."""

import pandas as pd
from tqdm import tqdm

from ft_dslr.logistic_regression import gradient_descent
from ft_dslr.logistic_regression.tools import denorm_thetas, normalise_df


def train_model(
    X: pd.DataFrame, y: pd.Series, learning_rate: float = 0.1, epoch: int = 1000
) -> dict:
    """
    Train the logistic regression model on the X and Y data.
    Parameters
    ----------
    X : The explanatory variables.
    y : The variables to predict
    learning_rate : The learning rate.
    epoch : The number of epochs to train the model.

    Returns
    -------
    A dictionary containing the trained model.
    """
    model = {}

    with tqdm(total=len(X.columns) * 4, desc="Training ", ncols=100) as pbar:
        for col in X.columns:
            _X = normalise_df(X[col].astype(float))
            _model = learn_multiple_y(
                _X,
                y,
                epoch=epoch,
                learning_rate=learning_rate,
                x_mean=X[col].astype(float).mean(),
                x_std=X[col].astype(float).std(),
                pbar=pbar,
            )
            model[col] = _model
    print("Done")
    return model


def learn_multiple_y(
    X: pd.Series,
    y: pd.Series,
    epoch: int,
    learning_rate: float,
    x_mean: float,
    x_std: float,
    pbar=None,
) -> dict:
    """
    Apply for each variable to predict, in the dataset, the gradient descent method.
    Parameters
    ----------
    X : The explanatory variables.
    y : The variables to predict.
    epoch : The number of epochs to train the model.
    learning_rate : The learning rate.
    x_mean : The mean of the X data.
    x_std : The standard deviation of the X data.
    pbar : The progress bar.

    Returns
    -------
    The trained model.
    """
    model = {}
    y = y.astype(int)
    params = y.unique()

    for param in params:
        _X = X.copy(deep=True)
        _y = y.replace(params, [1 if e == param else 0 for e in params])
        _model = gradient_descent(_X, _y, epoch=epoch, learning_rate=learning_rate)
        model[str(param)] = denorm_thetas(
            _model, x_mean, x_std, _y.astype(float).mean(), _y.astype(float).std()
        )
        if pbar is not None:
            pbar.update(1)

    return model
