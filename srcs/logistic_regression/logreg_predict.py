import pandas as pd
import json
import configparser
from tools.format_csv import config_drop_columns, format_csv, list_to_csv
import numpy as np
import argparse


def load_weights(weight_path: str):
    """
    Load the weights from the json file
    :param str, weight_path: the path to the weights.
    :return: the weights in a dictionary
    """
    with open(weight_path) as json_data:
        return json.load(json_data)


def load_config(config_path: str):
    """
    Load the configuration file
    :param str, config_path: the path to the configuration file
    :return: the configuration in a ConfigParser object
    """
    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read(config_path)
    return conf


def prepare_data(df_data: pd.DataFrame, config: configparser.ConfigParser):
    """
    Prepare the data for the prediction
    :param pd.DataFrame, df_data: the data
    :param configparser.ConfigParser, config: the configuration
    :return: the data and the target
    """
    df_data = config_drop_columns(df_data, config, verbose=False)
    X = df_data['Hogwarts House'].copy(deep=True)
    df_data.drop('Hogwarts House', axis=1, inplace=True)
    return df_data, X


def predict(df_data: pd.DataFrame, weights: dict):
    predict_list = []

    for index, row in df_data.iterrows():
        _total = {0: 0, 1: 0, 2: 0, 3: 0}
        for key, value in row.items():
            if key in weights:
                for house, thetas in weights[key].items():
                    if not pd.isnull(value) and isinstance(value, float):
                        _total[int(house)] += float(
                            1 / (1 + np.exp(-(float(thetas["theta0"]) + float(thetas["theta1"]) * float(value)))))
        predict_list.append((index, _total))
    return predict_list


def predict_house(predict_list: list):
    """
    Predict the house of the students
    :param list, predict_list: the list of predictions
    :return: Predicted labels
    """
    _y_pred = []
    for i, e in predict_list:
        _max = float('-inf')
        _house = None
        for house in e:
            if e[house] > _max:
                _max = e[house]
                _house = house
        _y_pred.append(int(_house))
    return _y_pred


def category_to_label(y: list, conf: configparser.ConfigParser):
    """
    Convert categories to string labels
    :param list, y: the binary labels
    :param conf: the configuration
    :return: predicted house labels
    """
    y_labels = []

    house_dict = dict(conf['REPLACE:Hogwarts House'])
    reversed_dict = {value: key for key, value in house_dict.items()}

    for i in y:
        y_labels.append(reversed_dict[str(i)])

    return y_labels


def logreg_predict(data_path: str, weight_path: str, config_path: str):
    """
    Predict the house of the students
    :param str, data_path: the path to the dataset
    :param str, weight_path: the path to the weights
    :param str, config_path: the path to the configuration file
    :return:
    """
    df_data = format_csv(data_path, config=config_path, norm_data=True)
    weights = load_weights(weight_path)
    config = load_config(config_path)

    df_data, X = prepare_data(df_data, config)
    predict_list = predict(df_data, weights)

    y_pred = predict_house(predict_list)

    pred_labels = category_to_label(y_pred, config)

    list_to_csv(pred_labels)


def options_parser():
    """Use to handle program parameters and options.
    """
    parser = argparse.ArgumentParser(
        prog='DSLR predict model',
        description='this program should be used to predict with a model of logistic regression',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('Dataset_file', type=str, nargs=1)
    parser.add_argument('Weights_file', type=str, nargs=1)
    parser.add_argument('-c', '--config', type=str, default='../../data/logistic.ini',
                        help='The config file.')
    return parser


if __name__ == '__main__':
    try:
        args = options_parser().parse_args()
        logreg_predict(args.Dataset_file[0], args.Weights_file[0], args.config)
    except Exception as e:
        print('[ERROR] The predict process failed')
        print(e)
