"""Tools ot normalise a dataframe."""

import pandas as pd


def normalise_df(df: pd.Series) -> pd.Series:
    """
    Normalise a dataframe to obtain a μ = 0 and σ = 1 with the Z-score.
    Parameters
    ----------
    df : The dataframe to be normalised.

    Returns
    -------
    The normalised dataframe.
    """
    df_cp = df.copy(deep=True)

    _mean = df_cp.mean()
    _std = df_cp.std()
    df_cp = df_cp.apply(lambda x: (x - _mean) / _std)

    return df_cp


def denormalize_thetas(model: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Denormalize the thetas.
    Parameters
    ----------
    model : The trained model.
    X : the explanatory variables.
    y : the variables to predict.

    Returns
    -------
    The denormalized model.
    """
    return model.apply(
        lambda x: [
            denormalize_t0(x.loc[0], x.loc[1], X[x.name], y),
            denormalize_t1(x.loc[1], X[x.name], y),
        ],
        axis=0,
    )


def denormalize_t0(theta0: float, theta1: float, X: pd.Series, y: pd.Series) -> float:
    """
    Denormalize the theta 0.
    Parameters
    ----------
    theta0 : First parameter.
    theta1 : Second parameter.
    X : The explanatory variables.
    y : The variables to predict.

    Returns
    -------
    The denormalized theta 0.
    """
    return y.mean() + y.std() * (theta0 - (theta1 * (X.mean() / X.std())))


def denormalize_t1(theta: float, X: pd.Series, y: pd.Series) -> float:
    """
    Denormalize the theta 1.
    Parameters
    ----------
    theta : The second theta.
    X : The explanatory variables.
    y : The variables to predict.

    Returns
    -------
    The denormalized theta 1.
    """
    return theta * (y.std() / X.std())
