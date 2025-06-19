"""Functions to train the logistic regression model."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logreg_predict import predict, predict_house
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from ft_dslr.logistic_regression.batch_selectors import mandatory_batch
from ft_dslr.logistic_regression.tools import (
    denormalize_thetas,
    normalise_df,
    sigmoid,
)


def train(
    X: pd.DataFrame,
    y: pd.Series,
    learning_rate: float = 0.1,
    epoch: int = 1000,
    X_test: pd.DataFrame = None,
    Y_test: pd.DataFrame = None,
    batch_selector: Callable[
        [pd.DataFrame, pd.Series], tuple
    ] = mandatory_batch,
) -> pd.DataFrame:
    """
    Train the logistic regression model.
    Parameters
    ----------
    X : The features.
    y : The target.
    learning_rate : The learning rate.
    epoch : Number of epochs.
    X_test : Test Features for the accuracy.
    Y_test : Test target for the accuracy,
    batch_selector : Function to select batches.

    Returns
    -------
    The model as a pd.DataFrame.
    """

    _X, _y, labels = prepare_data(X, y)
    thetas = build_thetas(X, labels)

    model = gradient_descent(
        _X,
        _y,
        thetas,
        labels,
        learning_rate,
        epoch,
        batch_selector=batch_selector,
        X_test=X_test,
        Y_test=Y_test,
    )

    model = denormalize_thetas(model, X, y)

    return model


def prepare_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Prepare data.
    Parameters
    ----------
    X : The features.
    y : The target.

    Returns
    -------
    Prepared X and y and labels.
    """
    data = {}
    y = y.astype(int)
    labels = y.unique().tolist()
    _X: pd.DataFrame = X.apply(
        lambda x: normalise_df(x.astype(float)), axis=0
    ).copy(deep=True)

    _y = pd.DataFrame()

    for value in labels:
        temp_df = (y == value).astype(int)
        data[value] = temp_df

    _y = pd.concat(data.values(), keys=data.keys())

    return _X, _y, labels


def build_thetas(X: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Init thetas.
    Parameters
    ----------
    X : The features.
    labels : The labels.

    Returns
    -------
    The initialized thetas.
    """
    data = {}

    for value in labels:
        temp_df = pd.DataFrame(0, index=[0, 1], columns=X.columns).astype(
            float
        )
        data[value] = temp_df

    result = pd.concat(data.values(), keys=data.keys())

    return result


def gradient_descent(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    thetas: pd.DataFrame,
    labels: list,
    learning_rate: float,
    epoch: int,
    batch_selector: Callable[
        [pd.DataFrame, pd.Series], tuple
    ] = mandatory_batch,
    X_test: pd.DataFrame = None,
    Y_test: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute thetas with the gradient descent.
    Parameters
    ----------
    X : The features.
    Y : The target.
    thetas : Initial thetas.
    labels : Labels.
    learning_rate : Learning rate.
    epoch : Number of epochs.
    batch_selector : Function to select batches.
    X_test : Test Features for the accuracy.
    Y_test : Test target for the accuracy.

    Returns
    -------
    The model as a pd.DataFrame.
    """

    acc = []

    for _ in tqdm(range(epoch)):
        _X, _Y = batch_selector(X, Y)

        for label in labels:
            thetas.loc[label] = _X.apply(
                lambda x: get_gradients(
                    x,
                    _Y[label],
                    thetas[x.name].loc[label, 0],
                    thetas[x.name].loc[label, 1],
                    len(_Y[label]),
                    learning_rate,
                ),
                axis=0,
            ).values

        if X_test is not None or Y_test is not None:
            acc.append(get_accurency(X_test, Y_test, thetas))

    if X_test is not None or Y_test is not None:
        plt.plot(acc)
    plt.show()

    return thetas


def get_gradients(
    X: pd.Series,
    Y: pd.Series,
    theta0: float,
    theta1: float,
    m: int,
    learning_rate: float,
) -> tuple[float, float]:
    """
    Compute the new gradient for a specific feature.
    Parameters
    ----------
    X : The feature.
    Y : The target.
    theta0 : First parameter to optimize.
    theta1 : Second parameter to optimize.
    m : Number of elements.
    learning_rate : The learning rate.

    Returns
    -------
    The thetas optimized.
    """

    _X = X.to_numpy()
    _Y = Y.to_numpy()

    z: np.ndarray = theta0 + theta1 * _X
    h: np.ndarray = sigmoid(z)

    error = h - _Y

    theta0 -= learning_rate * np.mean(error)
    theta1 -= learning_rate * np.dot(error, X) / m

    return theta0, theta1


def get_accurency(X_test, Y_test, model) -> float:
    y_pred_t = predict(X_test, model)
    y_pred_t = predict_house(y_pred_t)
    acc_test = accuracy_score(Y_test.astype(int).to_list(), y_pred_t) * 100

    return acc_test
