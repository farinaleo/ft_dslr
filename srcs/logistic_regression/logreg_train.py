from tools.format_csv import format_csv
from train import train_model
from tools.split import split_data
import argparse
import json


def save_model(model: dict, file_name: str) -> None:
    with open(file_name, 'w') as file:
        json.dump(model, file, indent=4)


def options_parser():
    """Use to handle program parameters and options.
    """
    parser = argparse.ArgumentParser(
        prog='DSLR train model',
        description='this program should be used to train a model of logistic regression able to predict this '
                    'hogwarts house of a student.',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('Train_file', type=str, nargs=1)
    parser.add_argument('-v', '--verbose', action='store_true', help='show the the results step by step in an GUI')
    parser.add_argument('-c', '--config', type=str, default='../../data/logistic.ini',
                        help='The config file.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1, help='The learning rate of the gradient '
                                                                               'descent.')
    parser.add_argument('-e', '--epoch', type=int, default=2500, help='The number of iterations of the '
                                                                      'gradient descent.')
    parser.add_argument('-y', '--y_col', type=str, default='Hogwarts House', help='The columns to predict.')
    return parser


if __name__ == '__main__':
    try:
        args = options_parser().parse_args()
        df = format_csv(args.Train_file[0], config=args.config, verbose=args.verbose)
        X_train, X_test, y_train, y_test = split_data(df, 0.2, None)
        model = train_model(X_train, y_train, args.learning_rate, args.epoch)
        save_model(model, 'model.json')
    except Exception as e:
        print('[ERROR] The training process failed')
        print(e)
