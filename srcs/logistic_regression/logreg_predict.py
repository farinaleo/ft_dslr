import pandas as pd
import json
import configparser
from .tools.format_csv import config_drop_columns
import numpy as np


def logreg_predict(data: str, weight: str):

    df_data = pd.read_csv(data)

    with open(weight) as json_data:
        weight = json.load(json_data)

    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read("data/logistic.ini")

    df_data = config_drop_columns(df_data, conf, verbose=False)
    df_data.drop('Hogwarts House', axis=1, inplace=True)

    df_data = df_data.dropna()

    predict_list = []

    for index, row in df_data.iterrows():
        predict = {}
        for key, value in row.items():
            if key in weight:
                for house, thetas in weight[key].items():
                    if house not in predict:
                        predict[house] = 0
                    print(thetas)
                    predict[house] = (1 / (1 + np.exp(-(thetas[0] + thetas[1] * value))))
        predict_list.append(predict)

    print(predict_list)
