"""Script used to plot a scatter plot."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from ft_dslr.tools import open_csv


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, verbose: bool = False) -> None:
    """
    Create a scatter plot of a dataframe column.
    Parameters
    ----------
    df : The source dataframe.
    x_col : The column name to use as abscissa.
    y_col : The column name to use as ordinate.
    verbose : Print additional information.

    Returns
    -------
    None.
    """
    if df["Hogwarts House"].isna().all():
        raise ValueError("No hogwarts houses found.")

    if verbose:
        print(f"Columns: {x_col}, {y_col}")

    font = {"weight": "light", "size": 7}
    plt.rc("font", **font)

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
        plt.scatter(house[0][x_col], house[0][y_col], s=1, color=house[1], alpha=0.5)

    plt.title(f"{x_col}, {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(["G", "R", "S", "H"], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def options_parser():
    """
    Used to handle command line options.
    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        prog="DSLR Scatter script.",
        description="this program should be used to plot scatters from the given dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("dataset", type=str, nargs=1)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all subjects as input.",
    )
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
        if not args.all:
            scatter_plot(df, "Defense Against the Dark Arts", "Astronomy", verbose=args.verbose)
        else:
            df_tmp = df.drop(columns="Index", inplace=False)
            cols = df_tmp.select_dtypes(include=["number"]).columns.tolist()
            for x in range(len(cols)):
                for y in range(len(cols)):
                    if x != y:
                        scatter_plot(df_tmp, cols[x], cols[y], verbose=args.verbose)

    except Exception as e:
        print("[ERROR] Could not open or use the file properly.")
        print(e)
        exit(1)
