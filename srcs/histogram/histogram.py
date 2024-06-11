import pandas as pd


def histogram(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

    # Group by Hogwarts House and calculate the mean of all courses
    return df.groupby('Hogwarts House')['Arithmancy'].mean()
