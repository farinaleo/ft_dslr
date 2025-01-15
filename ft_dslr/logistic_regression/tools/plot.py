from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(
    df: pd.DataFrame,
    col: str,
    house: Literal["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"] = "Gryffindor",
    beta_0: float = None,
    beta_1: float = None,
):
    """
    Plot the sigmoid curve with the labels.
    :param df: Source dataframe.
    :param col: Column name to consider as the y-axis.
    :param house: House use as label to identify with the sigmoid curve.
    :param beta_0: Coefficient of the sigmoid curve. 1 / (1 + np.exp((-(beta_1 * x + beta_0)))).
    :param beta_1: Coefficient of the sigmoid curve. 1 / (1 + np.exp((-(beta_1 * x + beta_0)))).
    :return: None
    """
    req_house = df[df["Hogwarts House"] == house]
    other_house = df[df["Hogwarts House"] != house]

    plt.scatter(other_house[col], [0] * len(other_house[col]), s=1, color="red")
    plt.scatter(req_house[col], [1] * len(req_house[col]), s=1, color="green")

    if beta_0 is not None and beta_1 is not None:
        x_list = np.arange(
            min(req_house[col].min(), other_house[col].min()),
            max(req_house[col].max(), other_house[col].max()) + 0.1,
            0.1,
        )
        y_list = [1 / (1 + np.exp((-(beta_1 * x + beta_0)))) for x in x_list]
        plt.plot(x_list, y_list, color="blue")

    plt.show()
