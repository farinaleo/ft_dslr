import pandas as pd
import configparser


def format_csv(path: str, config: str = None, norm_data: bool = False, verbose: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if config is not None:
            df = apply_config(config, df, verbose=verbose)
        if norm_data:
            df = normalise_csv(df)
        return df

    except Exception:
        raise ValueError('[ERROR] Some thing went wrong with csv and/or config file.')


def open_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def apply_config(config: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read(config)
    for col in conf['COLUMNS']:
        if not bool(conf['COLUMNS'].getboolean(col)) and col in df.columns:
            if verbose: print(f'[INFO] Drop {col}')
            df.drop(col, axis=1, inplace=True)
    return df


def normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['number']).columns.tolist():
        _mean = df[col].mean()
        _std = df[col].std()
        df[col] = df[col].apply(lambda x: (x - _mean) / _std)
    return df
