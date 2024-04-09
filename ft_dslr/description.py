import csv
import pandas
import numpy as np
import os


def describe_csv(dtf: pandas.DataFrame):



def describe_column(dtf: pandas.DataFrame, column: str):



def csv_to_dataframe(csv_file: str):



if __name__ == '__main__':
    try:
        dtf = csv_to_dataframe()
        describe_csv(dtf)
    except Exception as e:
        print('Error: the program can not read the csv file properly')
        print('Please make sure that the given path is correct and your datas are not corrupted')
        print('error caught: ', e)