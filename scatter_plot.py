import os
import json
import argparse
import subprocess


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
        with open('ft_dslr/setup.json', 'r') as file:
            data = json.load(file)
        data['scatter_path'] = file_path
        with open('ft_dslr/setup.json', 'w') as file:
            json.dump(data, file, indent=4)
        subprocess.call(['jupyter', 'execute', '--inplace', 'ft_dslr/scatter_plot.ipynb'])
        subprocess.call(['jupyter', 'notebook', 'ft_dslr/scatter_plot.ipynb'])
    except Exception as e:
        print('Error: the program can not read the csv file properly')
        print('Please make sure that the given path is correct and your datas are not corrupted')
        print('error caught: ', e)
        print('file path: ', file_path)
