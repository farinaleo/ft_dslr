import pandas as pd
import json
import configparser
from .tools.format_csv import config_drop_columns, format_csv
import numpy as np
from sklearn.metrics import accuracy_score

def logreg_predict(data: str, weight: str):
    df_data = format_csv(data, config="data/logistic.ini", norm_data=True)

    with open(weight) as json_data:
        weight = json.load(json_data)

    conf = configparser.ConfigParser()
    conf.optionxform = lambda option: option
    conf.read("data/logistic.ini")

    df_data = config_drop_columns(df_data, conf, verbose=False)
    X = df_data['Hogwarts House'].copy(deep=True)
    df_data.drop('Hogwarts House', axis=1, inplace=True)

    predict_list = []

    for index, row in df_data.iterrows():
        _total = {0: 0,
                  1: 0,
                  2: 0,
                  3: 0}
        for key, value in row.items():
            if key in weight:
                for house, thetas in weight[key].items():
                    if not pd.isnull(value) and type(value) == float:
                        _total[int(house)] += float(1 / (1 + np.exp(-(float(thetas["theta0"]) + float(thetas["theta1"]) * float(value)))))
        predict_list.append((index, _total))

    _y_pred = []
    _y_true = []


    print(len(predict_list))
    print(len(df_data.columns))
    print(len(df_data))
    print()
    cnt_true = 0
    for i, e in predict_list:
        _max = float('-inf')
        _house = None
        for house in e:
            if e[house] > _max:
                _max = e[house]
                _house = house
        print(f'predict house {_house} - {X.iloc[i]} | {int(_house) == int(X.iloc[i])}: {e}')
        if int(_house) == int(X.iloc[i]):
            cnt_true += 1
        _y_pred.append(int(_house))
        _y_true.append(int(X.iloc[i]))

    print()
    print(f'{cnt_true} / {len(predict_list)} | {(cnt_true / len(predict_list)) * 100} % de prediction')



    print(accuracy_score(_y_true, _y_pred))
