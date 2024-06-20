import pandas as pd
from tools.norm import normalise_df
from gradient_descent import gradient_descent
from tqdm import tqdm


def train_model(X: pd.DataFrame, y: pd.Series, learning_rate: float = 0.1, epoch: int = 1000) -> dict:
    model = {}

    with tqdm(total=len(X.columns) * 4, desc="Training ", ncols=100) as pbar:
        for col in X.columns:
            _X = normalise_df(X[col].astype(float))
            _model = learn_multiple_y(_X, y, epoch=epoch, learning_rate=learning_rate, pbar=pbar)
            model[col] = _model
    print('Done')
    return model


def learn_multiple_y(X: pd.Series, y: pd.Series, epoch: int, learning_rate: float, pbar=None) -> dict:
    model = {}
    y = y.astype(int)
    params = y.unique()

    for param in params:
        _X = X.copy(deep=True)
        _y = y.replace(params, [1 if e == param else 0 for e in params])
        model[str(param)] = gradient_descent(_X, _y, epoch=epoch, learning_rate=learning_rate)
        pbar.update(1)

    return model


