import sys

import matplotlib.pyplot as plt
import pandas as pd
from tools import open_csv


def pair_plot(df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Create a scatter subplot by plotting each column of the dataframe.
    :param df: the dataframe to plot.
    :param verbose: print additional information.
    :return: None
    """
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
    Create a scatter plot with the specified x and y columns.
    :param df: the dataframe to plot.
    :param x_cols: The list of all columns to use as x.
    :param y_cols: The list of all columns to use as y.
    :param fig: The figure to plot.
    :param x: coordinate of the x-axis.
    :param y: coordinate of the y-axis.
    :return: None.
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[ERROR] Usage: python pair_plot.py <file name>")
        exit(1)
    try:
        df = open_csv(sys.argv[1])
        pair_plot(df)
    except Exception as e:
        print("[ERROR] Could not open the file properly.")
        print(e)
        exit(1)
