import argparse

import matplotlib.pyplot as plt
import pandas as pd

from ft_dslr.tools import open_csv


def histogram(df: pd.DataFrame, subjects: list = None, verbose: bool = False) -> None:
    """
    Plot the histogram of the subjects in the dataset.
    :param df: the dataframe to plot.
    :param subjects: the list of subjects to plot.
    :param verbose: print additional information.
    :return: None
    """
    # Drop the index column
    df_tmp = df.drop(columns="Index", inplace=False)

    if df["Hogwarts House"].isna().all():
        raise ValueError("No hogwarts houses found.")

    # Get all subjects columns to plot
    subjects = (
        df_tmp.select_dtypes(include=["number"]).columns.tolist() if subjects is None else subjects
    )

    if verbose:
        print(f"Columns: {subjects}")

    # Set the font size
    font = {"weight": "light", "size": 7}
    plt.rc("font", **font)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(
        nrows=len(subjects), ncols=1, squeeze=False, figsize=(10, 5 * len(subjects))
    )

    # Plot each subject
    for i, subject in enumerate(subjects):
        plot_single_histogram(df_tmp, subject, axes[i, 0])

    # Set the title and show the plot
    plt.legend(["G", "R", "S", "H"], loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_single_histogram(df: pd.DataFrame, subject: str, ax: plt.Axes) -> None:
    """
    Plot a single histogram.

    :param df: the dataframe to plot.
    :param subject: the subject to plot.
    :param ax: the axis to plot.
    :return: None
    """
    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    colors = ["red", "blue", "green", "yellow"]

    for house, color in zip(houses, colors):
        house_df = df[df["Hogwarts House"] == house]
        ax.hist(house_df[subject].dropna(), bins=30, alpha=0.5, color=color, label=house)

    ax.set_ylabel(subject)
    ax.yaxis.label.set_rotation(45)


def options_parser():
    """Use to handle program parameters and options."""
    parser = argparse.ArgumentParser(
        prog="DSLR histogram script.",
        description="this program should be used to plot histogram from the given dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("dataset", type=str, nargs=1)
    parser.add_argument(
        "--columns", nargs="+", default=["Care of Magical Creatures"], help="Subject."
    )
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
        histogram(df, subjects=args.columns if not args.all else None, verbose=args.verbose)

    except Exception as e:
        print("[ERROR] Could not open or use the file properly.")
        print(e)
        exit(1)
