import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np


def plot(df: pd.DataFrame, col: str,
         house: Literal['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff'] = 'Gryffindor', beta_0: float = None,
         beta_1: float = None):
    req_house = df[df['Hogwarts House'] == house]
    other_house = df[df['Hogwarts House'] != house]

    plt.scatter(other_house[col], [0] * len(other_house[col]), s=1, color='red')
    plt.scatter(req_house[col], [1] * len(req_house[col]), s=1, color='green')

    if beta_0 is not None and beta_1 is not None:
        x_list = np.arange(min(req_house[col].min(), other_house[col].min()), max(req_house[col].max(), other_house[col].max()) + 0.1, 0.1)
        y_list = [1 / (1 + np.exp((-(beta_1 * x + beta_0)))) for x in x_list]
        plt.plot(x_list, y_list, color='blue')

    plt.show()
