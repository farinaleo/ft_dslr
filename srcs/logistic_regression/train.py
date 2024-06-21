import pandas as pd
from tools.norm import normalise_df, denorm_thetas
from gradient_descent import gradient_descent
from tqdm import tqdm


def train_model(X: pd.DataFrame, y: pd.Series, learning_rate: float = 0.1, epoch: int = 1000) -> dict:
    """
    Train a logistic regression model on the X and y data.
    :param X: The explanatory variables.
    :param y: The variable to predict.
    :param learning_rate: The learning rate applied to the gradient descent.
    :param epoch: The number of iterations to train the model.
    :return: A dictionary containing the training model for each explanatory variables.
    """
    model = {}

    with tqdm(total=len(X.columns) * 4, desc="Training ", ncols=100) as pbar:
        for col in X.columns:
            _X = normalise_df(X[col].astype(float))
            _model = learn_multiple_y(_X, y, epoch=epoch, learning_rate=learning_rate, x_mean=X[col].astype(float).mean(), x_std=X[col].astype(float).std(), pbar=pbar)
            model[col] = _model
    print('Done')
    return model


def learn_multiple_y(X: pd.Series, y: pd.Series, epoch: int, learning_rate: float, x_mean, x_std, pbar=None) -> dict:
    """
    Apply for each variable to predict, in the y dataset, the gradient descent.
    :param X: The explanatory variables.
    :param y: The variable to predict.
    :param epoch: The number of iterations to train the model.
    :param learning_rate: The learning rate applied to the gradient descent.
    :param pbar: The progress bar to update.
    :return: a dictionary containing the training model for each variable to predict.
    """
    model = {}
    y = y.astype(int)
    params = y.unique()

    for param in params:
        _X = X.copy(deep=True)
        _y = y.replace(params, [1 if e == param else 0 for e in params])
        _model = gradient_descent(_X, _y, epoch=epoch, learning_rate=learning_rate)
        model[str(param)] = denorm_thetas(_model, x_mean, x_std, _y.astype(float).mean(), _y.astype(float).std())
        if pbar is not None:
            pbar.update(1)

    return model
