import pandas as pd
import configparser


def format_csv(path: str, config: str = None, norm_data: bool = False, verbose: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if config is not None:
            df = apply_config(config, df, verbose=verbose)
        if norm_data:
            df = add_missing_values(df)
            df = normalise_csv(df)
        return df

    except Exception as e:
        raise ValueError(f'[ERROR] Some thing went wrong with csv and/or config file.\n{e}')


def open_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def apply_config(config: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read(config)
    df = config_drop_columns(df, conf, verbose=verbose)
    df = config_replace_values(df, conf, verbose=verbose)
    return df



def config_drop_columns(df: pd, config:configparser.ConfigParser, verbose:bool=False) -> pd.DataFrame:
    for col in config['COLUMNS']:
        if not bool(config['COLUMNS'].getboolean(col)) and col in df.columns:
            if verbose: print(f'[INFO] Drop {col}')
            df.drop(col, axis=1, inplace=True)
    return df


def config_replace_values(df: pd, config:configparser.ConfigParser, verbose:bool=False) -> pd.DataFrame:
    for section in config.sections():
        if section.startswith('REPLACE:'):
            sub_section = section.replace('REPLACE:', '')
            if sub_section in df.columns:
                for var in config[section]:
                    if verbose: print(f'[INFO] replace {var} by {config[section][var]} in {sub_section}')
                    df[sub_section] = df[sub_section].replace(var, config[section][var])
    return df


def add_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['number']).columns.tolist():
        _mean = df[col].mean()
        df[col] = df[col].fillna(_mean)
    return df


def normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['number']).columns.tolist():
        _mean = df[col].mean()
        _std = df[col].std()
        df[col] = df[col].apply(lambda x: (x - _mean) / _std)
    return df
