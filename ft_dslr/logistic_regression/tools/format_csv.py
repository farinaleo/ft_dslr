"""Tools to format and apply configurations on csv."""

import configparser
import os

import pandas as pd


def format_csv(
    path: str,
    config: str = None,
    add_missing: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Format and apply configurations on csv files.
    Parameters
    ----------
    path : The file path.
    config : The configuration path.
    add_missing : Option to replace missing values. (Use the Mean Imputation).
    verbose : Print additional information.

    Returns
    -------
    The formated dataframe.
    """
    try:
        df = pd.read_csv(path)
        if config is not None:
            df = apply_config(config, df, verbose=verbose)
        if add_missing:
            df = add_missing_values(df)
        return df

    except Exception as e:
        raise ValueError(
            f"[ERROR] Some thing went wrong with csv and/or config file.\n{e}"
        )


def apply_config(
    config: str, df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """
    Apply configurations to dataframe.
    Parameters
    ----------
    config : The configuration path.
    df : The dataframe to be formatted.
    verbose : Print additional information.

    Returns
    -------
    The formatted dataframe.
    """
    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read(config)
    df = config_drop_columns(df, conf, verbose=verbose)
    df = config_replace_values(df, conf, verbose=verbose)
    return df


def config_drop_columns(
    df: pd, config: configparser.ConfigParser, verbose: bool = False
) -> pd.DataFrame:
    """
    Drop specified columns from dataframe.
    Parameters
    ----------
    df : The dataframe to be formatted.
    config : The configuration as a configparser.ConfigParser object.
    verbose : Print additional information.

    Returns
    -------
    The formatted dataframe.
    """
    for col in config["COLUMNS"]:
        if not bool(config["COLUMNS"].getboolean(col)) and col in df.columns:
            if verbose:
                print(f"[INFO] Drop {col}")
            df.drop(col, axis=1, inplace=True)
    return df


def config_replace_values(
    df: pd, config: configparser.ConfigParser, verbose: bool = False
) -> pd.DataFrame:
    """
    Replace specified values, in the configuration, in dataframe.
    Parameters
    ----------
    df : The dataframe to be formatted.
    config : The configuration as a configparser.ConfigParser object.
    verbose : Print additional information.

    Returns
    -------
    The formatted dataframe.
    """
    for section in config.sections():
        if section.startswith("REPLACE:"):
            sub_section = section.replace("REPLACE:", "")
            if sub_section in df.columns:
                for var in config[section]:
                    if verbose:
                        print(
                            f"[INFO] replace {var} by ",
                            f"{config[section][var]} in {sub_section}"
                        )
                    df[sub_section] = df[sub_section].replace(
                        var, config[section][var]
                    )
    return df


def add_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing values to dataframe with the Mean Imputation.
    Parameters
    ----------
    df : The dataframe to be formatted.

    Returns
    -------
    the formatted dataframe.
    """
    for col in df.select_dtypes(include=["number"]).columns.tolist():
        _mean = df[col].mean()
        df[col] = df[col].fillna(_mean)
    return df


def normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the dataframe with the Z-score.
    Parameters
    ----------
    df : The dataframe to be formatted.

    Returns
    -------
    The formatted dataframe.
    """
    for col in df.select_dtypes(include=["number"]).columns.tolist():
        _mean = df[col].mean()
        _std = df[col].std()
        df[col] = df[col].apply(lambda x: (x - _mean) / _std)
    return df


def list_to_csv(y_list: list, dest_path: str):
    """
    Convert a list to a csv file and save it.
    Parameters
    ----------
    y_list : The list to convert.
    dest_path : The destination path.

    Returns
    -------
    None
    """
    df = pd.DataFrame(y_list, columns=["Hogwarts House"])
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Index"}, inplace=True)
    df.to_csv(os.path.join(dest_path, "houses.csv"), index=False)
