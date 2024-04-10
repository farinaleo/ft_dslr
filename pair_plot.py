import os
import argparse
from ft_dslr.description import csv_to_dataframe
from ft_dslr.pair_plot import *


def args_parser():
    """Use to handle program parameters and options."""
    parser = argparse.ArgumentParser(
        prog='Pair plot',
        description='this program should be used to make a pair plot from csv file to understand datas',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('Input_file', type=str, nargs=1)
    return parser


if __name__ == '__main__':
    file_path = ''
    try:
        args = args_parser().parse_args()
        file_path = os.path.abspath(os.path.dirname(__file__)) + '/' + args.Input_file[0]
        dtf = csv_to_dataframe(file_path)
        pair_plot(dtf)
    except Exception as e:
        print('Error: the program can not read the csv file properly')
        print('Please make sure that the given path is correct and your datas are not corrupted')
        print('error caught: ', e)
        print('file path: ', file_path)
