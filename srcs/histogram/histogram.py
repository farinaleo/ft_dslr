import pandas as pd
from tools import open_csv
import sys
import matplotlib.pyplot as plt


def histogram(df: pd.DataFrame, subjects: list = None, verbose: bool = False) -> None:
    """
    Plot the histogram of the subjects in the dataset.
    :param df: the dataframe to plot.
    :param subjects: the list of subjects to plot.
    :param verbose: print additional information.
    :return: None
    """
    # Drop the index column
    df_tmp = df.drop(columns='Index', inplace=False)

    # Get all subjects columns to plot
    subjects = df_tmp.select_dtypes(include=['number']).columns.tolist() if subjects is None else subjects

    if verbose:
        print(f'Columns: {subjects}')

    # Set the font size
    font = {'weight': 'light', 'size': 7}
    plt.rc('font', **font)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=len(subjects), ncols=1, squeeze=False, figsize=(5, 30))

    # Plot each subject
    for i, subject in enumerate(subjects):
        plot_single_histogram(df_tmp, subject, axes[i, 0])

    # Set the title and show the plot
    plt.legend(['G', 'R', 'S', 'H'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_single_histogram(df: pd.DataFrame, subject: str, ax: plt.Axes) -> None:
    """
    Plot a single histogram.

    :param df: the dataframe to plot.
    :param subject: the subject to plot.
    :param ax: the axis to plot.
    :return: None
    """
    houses = ['Gryffindor', 'Ravenclaw', 'Slytherin', 'Hufflepuff']
    colors = ['red', 'blue', 'green', 'yellow']

    for house, color in zip(houses, colors):
        house_df = df[df['Hogwarts House'] == house]
        ax.hist(house_df[subject].dropna(), bins=30, alpha=0.5, color=color, label=house)

    ax.set_ylabel(subject)
    ax.yaxis.label.set_rotation(45)


if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('[ERROR] Usage: python histogram.py <file name> to get the answer or python histogram.py <file '
              'name> --all to visualise all features plot with each other.')
        exit(1)
    try:
        df = open_csv(sys.argv[1])
        if len(sys.argv) == 2:
            histogram(df, ['Care of Magical Creatures'], verbose=True)
        elif sys.argv[2] == '--all':
            histogram(df, verbose=True)
        else:
            raise ValueError('[ERROR] Usage: option not handled.')

    except Exception as e:
        print('[ERROR] Could not open the file properly.')
        print(e)
        exit(1)
