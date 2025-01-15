"""Custom description of a dataset."""

import argparse
import math

import pandas as pd

from ft_dslr.tools import open_csv


def describe(df: pd.DataFrame, drop: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Describe the given dataframe.
    Parameters
    ----------
    df : The dataframe to describe.
    drop :  Drop empty columns.
    verbose : Print additional information.

    Returns
    -------
    The description as a dataframe.
    """
    description = pd.DataFrame()

    if drop:
        df = df.dropna(axis=1, how="all")

    df_tmp = df.drop(columns="Index", inplace=False)
    columns_num = df_tmp.select_dtypes(include=["number"]).columns.tolist()
    df_tmp = df_tmp[columns_num]
    df_tmp = df_tmp.astype(float)

    description["Info"] = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    for col in df_tmp.columns:
        if verbose:
            print(f"Column used: {col}")
        description[col] = describe_column(df_tmp, col)

    return description


def describe_column(df: pd.DataFrame, column: str) -> list:
    """
    Describe a specific column from the dataframe.
    Parameters
    ----------
    df : The source dataframe.
    column : The column to describe.

    Returns
    -------
    A list of elements used to describe the column.
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
    Parameters
    ----------
    df : The source dataframe.
    column : The column to describe.

    Returns
    -------
    The mean of the column.
    """
    sum_v = 0

    for _, value in df[column].items():
        sum_v += float(value)

    return float(sum_v / len(df[column]))


def compute_std(df: pd.DataFrame, column: str, mean: float) -> float:
    """
    Compute the standard deviation of the given column.
    Parameters
    ----------
    df : The source dataframe.
    column : The column to describe.
    mean : The mean of the column.

    Returns
    -------
    The standard deviation of the column.
    """
    sum_v = 0

    for _, value in df[column].items():
        sum_v += (float(value) - mean) ** 2

    return float(math.sqrt(sum_v / len(df[column])))


def find_min_max(df: pd.DataFrame, column: str) -> tuple:
    """
    Find the minimum and maximum values of the given column.
    Parameters
    ----------
    df : The source dataframe.
    column : The column to describe.

    Returns
    -------
    A tuple containing the minimum and maximum values of the column.
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
    Parameters
    ----------
    df : The source dataframe.
    column : The column to describe.
    q : The quantile to find.

    Returns
    -------
    The quantile of the column.
    """
    q_idx = q * len(df[column])
    if q_idx.is_integer():
        q_v = df.loc[q_idx, column]
    elif q_idx - int(q_idx) < 0.5:
        q_v = df.loc[int(q_idx), column]
    else:
        q_v = df.loc[math.ceil(q_idx), column]
    return q_v


def options_parser():
    """
    Used to handle command line options.
    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        prog="DSLR describe dataset",
        description="this program should be used to describe a dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("dataset", type=str, nargs=1)
    parser.add_argument("-d", "--drop", action="store_true", help="Drop empty columns.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print additional information about the process.",
    )

    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()
        df = open_csv(
            args.dataset[0],
        )
        description = describe(df, drop=args.drop, verbose=args.verbose)
        print(description)
        print("\n[INFO] To see the entire description, use the jupyter notebook.")
    except Exception as e:
        print("[ERROR] Could not open the file properly.")
        print(f"[ERROR] Error: {e}")
        exit(1)
