import pandas as pd
from tools import open_csv
import sys
import matplotlib.pyplot as plt


def scatter_plot(df: pd.DataFrame, y_cols: list = None, verbose: bool = False) -> None:
    """
    Create a scatter subplot by plotting each column of the dataframe.
    :param df: the dataframe to plot.
    :param y_cols: the list of columns to use as y. If not specified use all columns.
    :param verbose: print additional information.
    :return: None
    """
    df_tmp = df.drop(columns='Index', inplace=False)
    x_cols = df_tmp.select_dtypes(include=['number']).columns.tolist()
    y_cols = x_cols if y_cols is None else y_cols

    if verbose: print(f'Columns: {x_cols}')

    font = {'weight': 'light', 'size': 7}
    plt.rc('font', **font)

    x_size, y_size = len(x_cols), len(y_cols)
    fig, axes = plt.subplots(nrows=y_size, ncols=x_size, squeeze=False, sharex='col', sharey='row', figsize=(20, 20))

    for x in range(x_size):
        for y in range(y_size):
            plot_single_scatter(df_tmp, x_cols, y_cols, axes[y, x], x, y)

    plt.show()


def plot_single_scatter(df: pd.DataFrame, x_cols: list, y_cols: list, fig: plt.Axes, x: int, y: int) -> None:
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
    _gryffindor = df[df['Hogwarts House'] == 'Gryffindor']
    _hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff']
    _slytherin = df[df['Hogwarts House'] == 'Slytherin']
    _ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw']

    houses = [(_gryffindor, 'red'), (_ravenclaw, 'blue'), (_slytherin, 'green'), (_hufflepuff, 'yellow')]

    for house in houses:
        fig.scatter(house[0][x_cols[x]], house[0][y_cols[y]], s=1, color=house[1], alpha=0.5)

    if y == len(y_cols) - 1:
        fig.set_xlabel(x_cols[x].replace(' ', '\n'), rotation=45)
    if x == 0:
        fig.set_ylabel(y_cols[y].replace(' ', '\n'))

    fig.tick_params(labelrotation=45.0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[ERROR] Usage: python scatter_plot.py <file name> <y_col1> ... <y_coln>')
        exit(1)
    try:
        df = open_csv(sys.argv[1])
        scatter_plot(df, y_cols=sys.argv[2:] if len(sys.argv) > 2 else None, verbose=True)
    except Exception as e:
        print('[ERROR] Could not open the file properly.')
        print(e)
        exit(1)
