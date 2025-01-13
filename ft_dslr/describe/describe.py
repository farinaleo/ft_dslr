import math
import sys

import pandas as pd
from tools import open_csv


def describe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Describe the given dataframe.
    :param df: The dataframe to describe.
    :param verbose: print additional information about the process.
    :return: The description as a pandas dataframe.
    """
    description = pd.DataFrame()

    df_tmp = df.drop(columns="Index", inplace=False)
    columns_num = df_tmp.select_dtypes(include=["number"]).columns.tolist()
    df_tmp = df_tmp[columns_num]
    df_tmp = df_tmp.astype(float)

    description["Info"] = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    for col in df_tmp.columns:
        if verbose:
            print(f"Column: {col}")
        description[col] = describe_column(df_tmp, col)

    return description


def describe_column(df: pd.DataFrame, column: str) -> list:
    """
    Describe a specific column in the dataframe.
    :param df: the dataframe to describe.
    :param column: the column to describe.
    :return: a list of elements used to describe the column.
    """
    df_tmp = pd.DataFrame(df[column].copy())
    df_tmp.dropna(inplace=True)
    df_tmp.sort_values(by=column, ascending=True, inplace=True)
    df_tmp.reset_index(drop=True, inplace=True)

    count = len(df_tmp[column])
    mean = compute_mean(df_tmp, column)
    std = compute_std(df_tmp, column, mean)
    min_v, max_v = find_min_max(df_tmp, column)
    q25 = compute_quantile(df_tmp, column, 0.25)
    q50 = compute_quantile(df_tmp, column, 0.5)
    q75 = compute_quantile(df_tmp, column, 0.75)

    return [count, mean, std, min_v, q25, q50, q75, max_v]


def compute_mean(df: pd.DataFrame, column: str) -> float:
    """
    Compute the mean of the given column.
    :param df: the dataframe to describe.
    :param column: the column to describe.
    :return: the mean of the given column.
    """
    sum_v = 0

    for _, value in df[column].items():
        sum_v += float(value)

    return float(sum_v / len(df[column]))


def compute_std(df: pd.DataFrame, column: str, mean: float) -> float:
    """
    Compute the standard deviation of the given column.
    :param df: the dataframe to describe.
    :param column: th column to describe.
    :param mean: the mean of the column.
    :return: the standard deviation of the given column.
    """
    sum_v = 0

    for _, value in df[column].items():
        sum_v += (float(value) - mean) ** 2

    return float(math.sqrt(sum_v / len(df[column])))


def find_min_max(df: pd.DataFrame, column: str) -> tuple:
    """
    Find the min and the ma of the given column.
    :param df: the dataframe to describe.
    :param column: the column to describe.
    :return: a tuple of the min and the max of the given column.
    """
    min_v = float("inf")
    max_v = float("-inf")

    for _, value in df[column].items():
        if value > max_v:
            max_v = float(value)
        if value < min_v:
            min_v = float(value)

    return min_v, max_v


def compute_quantile(df: pd.DataFrame, column: str, q: float) -> tuple:
    """
    Compute the quantile of the given column, with respect to the given value.
    :param df: the dataframe to describe.
    :param column: the column to describe.
    :param q: the quantile to find.
    :return: the quantile of the given column.
    """
    q_idx = q * len(df[column])
    if q_idx.is_integer():
        q_v = df.loc[q_idx, column]
    elif q_idx - int(q_idx) < 0.5:
        q_v = df.loc[int(q_idx), column]
    else:
        q_v = df.loc[math.ceil(q_idx), column]
    return q_v


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[ERROR] Usage:python describe.py <file to describe>")
        exit(1)
    try:
        df = open_csv(sys.argv[1])
        description = describe(df)
        print(description)
        print("\n[INFO] To see the entire description, use the jupyter notebook.")
    except Exception:
        print("[ERROR] Could not open the file properly.")
        exit(1)
