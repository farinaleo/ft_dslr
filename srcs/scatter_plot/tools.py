import pandas as pd


def open_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return df