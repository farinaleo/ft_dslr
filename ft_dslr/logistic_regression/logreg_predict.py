"""Script used to predict value from a dataframe with a pre-computed logistic regression model."""

import argparse
import configparser

import numpy as np
import pandas as pd

from ft_dslr.logistic_regression.tools import (
    config_drop_columns,
    format_csv,
    list_to_csv,
)


def load_model(model_path: str):
    """
    Load model from a file.
    Parameters
    ----------
    model_path : The file path.

    Returns
    -------
    The loaded model.
    """
    return pd.read_csv(model_path, index_col=[0, 1])


def load_config(config_path: str):
    """
    Load the configuration from a file.
    Parameters
    ----------
    config_path : The file path.

    Returns
    -------
    The loaded configuration.
    """
    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read(config_path)
    return conf


def prepare_data(df_data: pd.DataFrame, config: configparser.ConfigParser):
    """
    Prepare the data for the prediction.
    Parameters
    ----------
    df_data : The date to prepare.
    config : The configuration to apply.

    Returns
    -------
    The prepared data.
    """
    df_data = config_drop_columns(df_data, config, verbose=False)
    X = df_data["Hogwarts House"].copy(deep=True)
    df_data.drop("Hogwarts House", axis=1, inplace=True)
    return df_data, X


def predict(df_data: pd.DataFrame, model: pd.DataFrame) -> list:
    """
    Apply the computed model to predict each house of the dataframe.
    Parameters
    ----------
    df_data : The source dataframe to predict.
    model : A dataframe containing the model.

    Returns
    -------

    """
    predict_list = []

    for index, row in df_data.iterrows():
        _total = {0: 0, 1: 0, 2: 0, 3: 0}
        for col, value in row.items():
            for house in list(model.index.get_level_values(0).unique()):
                if not pd.isnull(value) and isinstance(value, float):
                    _total[int(house)] += float(
                        1
                        / (
                            1
                            + np.exp(
                                -(
                                    float(model[col].loc[house, 0])
                                    + float(model[col].loc[house, 1]) * float(value)
                                )
                            )
                        )
                    )
        predict_list.append((index, _total))
    return predict_list


def predict_house(predict_list: list):
    """
    Predict each house of the dataframe.
    Parameters
    ----------
    predict_list : The list of predictions.

    Returns
    -------
    Predicted labels.
    """
    _y_pred = []
    for i, e in predict_list:
        _max = float("-inf")
        _house = None
        for house in e:
            if e[house] > _max:
                _max = e[house]
                _house = house
        _y_pred.append(int(_house))
    return _y_pred


def category_to_label(y: list, conf: configparser.ConfigParser):
    """
    Convert categories to string labels.
    Parameters
    ----------
    y : The binary labels.
    conf : The configuration to apply.

    Returns
    -------
    The predicted house labels.
    """
    y_labels = []

    house_dict = dict(conf["REPLACE:Hogwarts House"])
    reversed_dict = {value: key for key, value in house_dict.items()}

    for i in y:
        y_labels.append(reversed_dict[str(i)])

    return y_labels


def logreg_predict(data_path: str, model_path: str, config_path: str, dest_path: str):
    """
    Predict the house of each student.
    Parameters
    ----------
    data_path : The path of the csv source.
    model_path : The model path.
    config_path : The configuration path.
    dest_path : The destination path.

    Returns
    -------
    None
    """
    df_data = format_csv(data_path, config=config_path, norm_data=False)
    model = load_model(model_path)
    config = load_config(config_path)

    df_data, X = prepare_data(df_data, config)
    predict_list = predict(df_data, model)

    y_pred = predict_house(predict_list)

    pred_labels = category_to_label(y_pred, config)

    list_to_csv(pred_labels, dest_path)


def options_parser():
    """
    Used to handle command line options.
    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        prog="DSLR predict model",
        description="this program should be used to predict with a model of logistic regression",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("Dataset_file", type=str, nargs=1)
    parser.add_argument("Weights_file", type=str, nargs=1)
    parser.add_argument(
        "-c", "--config", type=str, default="models/logistic.ini", help="The config file."
    )
    parser.add_argument(
        "-d",
        "--dest",
        type=str,
        default="data/",
        help="Destination path, location where to save prediction.",
    )
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()
        logreg_predict(args.Dataset_file[0], args.Weights_file[0], args.config, args.dest)
    except Exception as e:
        print("[ERROR] The predict process failed")
        print(e)
