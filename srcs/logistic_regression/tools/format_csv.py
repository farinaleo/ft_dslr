import pandas as pd
import configparser



def format_csv(
    path: str, config: str = None, norm_data: bool = False, verbose: bool = False
) -> pd.DataFrame:
    """
    Format a CSV file to be used for a Logistic Regression. If a config file is specified, it will be used
    to remove Columns and replace values.
    :param path: The path of the file.
    :param config: The path of the config file.
    :param norm_data: Option to normalize the data. (apply the Z-score and replace missing values with Mean Imputation)
    :param verbose: Print addition information.
    :return: The formatted df.DataFrame.
    """
    try:
        df = pd.read_csv(path)
        if config is not None:
            df = apply_config(config, df, verbose=verbose)
        if norm_data:
            df = add_missing_values(df)
            df = normalise_csv(df)
        return df

    except Exception as e:
        raise ValueError(f"[ERROR] Some thing went wrong with csv and/or config file.\n{e}")


def open_csv(path: str) -> pd.DataFrame:
    """
    Open a CSV file to be used convert as a pandas DataFrame.
    :param path: The path of the file.
    :return: The csv as a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df


def apply_config(config: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Apply a config file to the given data frame.
    :param config: the path of the config file.
    :param df: the data frame.
    :param verbose: Print addition information.
    :return: The formatted df.
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
    Drop columns specified in the config file.
    :param df: the data frame.
    :param config: the config file as a ConfigParser object.
    :param verbose: Print addition information.
    :return: the formatted data frame.
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
    Replace values specified in the config file.
    :param df: the data frame.
    :param config: the config file as a ConfigParser object.
    :param verbose: Print additional information.
    :return: the formatted data frame.
    """
    for section in config.sections():
        if section.startswith("REPLACE:"):
            sub_section = section.replace("REPLACE:", "")
            if sub_section in df.columns:
                for var in config[section]:
                    if verbose:
                        print(f"[INFO] replace {var} by {config[section][var]} in {sub_section}")
                    df[sub_section] = df[sub_section].replace(var, config[section][var])
    return df


def add_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing with Mean Imputation.
    :param df: the data frame.
    :return: the formatted data frame.
    """
    for col in df.select_dtypes(include=["number"]).columns.tolist():
        _mean = df[col].mean()
        df[col] = df[col].fillna(_mean)
    return df


def normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the data frame with the Z-score.
    :param df: the data frame.
    :return: the formatted data frame.
    """
    for col in df.select_dtypes(include=["number"]).columns.tolist():
        _mean = df[col].mean()
        _std = df[col].std()
        df[col] = df[col].apply(lambda x: (x - _mean) / _std)
    return df


def list_to_csv(y_list: list):
    """
    Convert a list to a CSV file and save it.
    :param y_list: the list to convert.
    :return:
    """
    df = pd.DataFrame(y_list, columns=["Hogwarts House"])
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Index"}, inplace=True)
    df.to_csv("../../data/houses.csv", index=False)
