"""Functions to train the logistic regression model."""

from typing import Callable

import pandas as pd
from tqdm import tqdm

from ft_dslr.logistic_regression import gradient_descent
from ft_dslr.logistic_regression.batch_selectors import mandatory_batch
from ft_dslr.logistic_regression.tools import denormalize_thetas, normalise_df


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    learning_rate: float = 0.1,
    epoch: int = 1000,
    batch_selector: Callable[[pd.DataFrame, pd.Series], tuple] = mandatory_batch,
) -> pd.DataFrame:
    """
    Train the logistic regression model on the X and Y data.
    Parameters
    ----------
    X : The explanatory variables.
    y : The variables to predict
    learning_rate : The learning rate.
    epoch : The number of epochs to train the model.
    batch_selector : The batch selector function.

    Returns
    -------
    A dataframe containing the model.
    """
    models = {}

    with tqdm(total=len(X.columns) * 4, desc="Training ", ncols=100) as pbar:

        models = learn_multiple_y(
            X,
            y,
            epoch=epoch,
            learning_rate=learning_rate,
            pbar=pbar,
            batch_selector=batch_selector,
        )

    model_df = pd.concat(models.values(), keys=models.keys())
    print("Done")
    return model_df


def learn_multiple_y(
    X: pd.DataFrame,
    y: pd.Series,
    epoch: int,
    learning_rate: float,
    pbar=None,
    batch_selector: Callable[[pd.DataFrame, pd.Series], tuple] = mandatory_batch,
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
    batch_selector : The batch selector function.

    Returns
    -------
    The trained model.
    """
    model = {}
    y = y.astype(int)
    params = y.unique()

    for param in params:
        _X = X.apply(lambda x: normalise_df(x.astype(float)), axis=0).copy(deep=True)
        _y = y.replace(params, [1 if e == param else 0 for e in params])
        _model = gradient_descent(
            _X, _y, epoch=epoch, learning_rate=learning_rate, batch_selector=batch_selector
        )

        print(_model)
        model[str(param)] = denormalize_thetas(_model, X, y)
        print(model[str(param)])
        if pbar is not None:
            pbar.update(len(X.columns))

    return model
