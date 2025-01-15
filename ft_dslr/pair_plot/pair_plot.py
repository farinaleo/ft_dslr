"""Script to plot a pair plot of a dataframe."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from ft_dslr.tools import open_csv


def pair_plot(df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Create a pair plot of a dataframe.
    Parameters
    ----------
    df : The source dataframe
    verbose : Print additional information.

    Returns
    -------
    None.
    """
    if df["Hogwarts House"].isna().all():
        raise ValueError("No hogwarts houses found.")

    df_tmp = df.drop(columns="Index", inplace=False)
    x_col = df_tmp.select_dtypes(include=["number"]).columns.tolist()
    y_col = x_col

    if verbose:
        print(f"Columns: {x_col}")

    font = {"weight": "light", "size": 7}
    plt.rc("font", **font)

    x_size, y_size = len(x_col), len(y_col)
    fig, axes = plt.subplots(
        nrows=y_size, ncols=x_size, squeeze=False, sharex=False, sharey=False, figsize=(20, 20)
    )

    for x in range(x_size):
        for y in range(y_size):
            plot_single_graph(df_tmp, x_col, y_col, axes[y, x], x, y)

    plt.legend(["G", "R", "S", "H"], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_single_graph(
    df: pd.DataFrame, x_cols: list, y_cols: list, fig: plt.Axes, x: int, y: int
) -> None:
    """
    Plot a single graph of a dataframe.
    Parameters
    ----------
    df : The source dataframe
    x_cols : The columns to use as abscissa.
    y_cols : The columns to use as ordinates.
    fig : The matplotlib figure.
    x : Coordinate of the x-axis.
    y : Coordinate of the y-axis.

    Returns
    -------
    None
    """
    _gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    _hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    _slytherin = df[df["Hogwarts House"] == "Slytherin"]
    _ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]

    houses = [
        (_gryffindor, "red"),
        (_ravenclaw, "blue"),
        (_slytherin, "green"),
        (_hufflepuff, "yellow"),
    ]

    for house in houses:
        if x != y:
            fig.scatter(house[0][x_cols[x]], house[0][y_cols[y]], s=1, color=house[1], alpha=0.5)
        else:
            fig.hist(house[0][x_cols[x]], color=house[1], alpha=0.5)

    if y == len(y_cols) - 1:
        fig.set_xlabel(x_cols[x].replace(" ", "\n"), rotation=45)
    if x == 0:
        fig.set_ylabel(y_cols[y].replace(" ", "\n"))

    fig.tick_params(labelrotation=45.0)


def options_parser():
    """
    Used to handle command line options.
    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        prog="DSLR Pair plot script",
        description="this program should be used to produce pair plot of a dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("dataset", type=str, nargs=1)
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
        df = open_csv(args.dataset[0])
        pair_plot(df, args.verbose)
    except Exception as e:
        print("[ERROR] Could not open the file properly.")
        print("[ERROR] error: ", e)
        exit(1)
