import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tools import open_csv


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, verbose: bool = False) -> None:
    """
    Create a scatter plot.
    :param df: the dataframe to plot.
    :param x_col: the name of the x column to plot.
    :param y_col: the name of the y column to plot.
    :param verbose: print additional information.
    :return: None
    """

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


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(
            "[ERROR] Usage: python scatter_plot.py <file name> to get the answer or python scatter_plot.py <file "
            "name> --all to visualise all features plot with each other."
        )
        exit(1)
    try:
        df = open_csv(sys.argv[1])
        if len(sys.argv) == 2:
            scatter_plot(df, "Defense Against the Dark Arts", "Astronomy", verbose=True)
        elif sys.argv[2] == "--all":
            df_tmp = df.drop(columns="Index", inplace=False)
            cols = df_tmp.select_dtypes(include=["number"]).columns.tolist()
            for x in range(len(cols)):
                for y in range(len(cols)):
                    if x != y:
                        scatter_plot(df_tmp, cols[x], cols[y], verbose=True)
        else:
            raise ValueError("[ERROR] Usage: option not handled.")

    except Exception as e:
        print("[ERROR] Could not open the file properly.")
        print(e)
        exit(1)
