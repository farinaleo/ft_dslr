"""Script used to train the logistic regression model."""

import argparse
import os
import random

import pandas as pd
from sklearn.metrics import accuracy_score

from ft_dslr.logistic_regression.batch_selectors import (
    mandatory_batch,
    mini_batch,
    stochastic_batch,
)
from ft_dslr.logistic_regression.gradient_descent import train
from ft_dslr.logistic_regression.logreg_predict import predict, predict_house
from ft_dslr.logistic_regression.tools import format_csv, save_model, split_data

BATCH_SELECTOR = {
    "mandatory": mandatory_batch,
    "stochastic": stochastic_batch,
    "mini": mini_batch,
}


def options_parser():
    """
    Used to handle command line options.
    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        prog="DSLR train model",
        description="this program should be used to train a model of logistic regression able to predict the "
        "hogwarts house of a student.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("Train_file", type=str, nargs=1)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show the the results step by step in an GUI"
    )
    parser.add_argument(
        "-a",
        "--accuracy",
        action="store_true",
        help="Compute only the accuracy score and not launch the training process.",
    )
    parser.add_argument(
        "-c", "--config", type=str, default="models/logistic.ini", help="The config file."
    )
    parser.add_argument("-m", "--model", type=str, default="model.csv", help="The model file.")
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.1,
        help="The learning rate of the gradient descent.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=2500,
        help="The number of iterations of the gradient descent.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Ratio of data used for the validation dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Seed used to reproduce a random split.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="mandatory",
        choices=BATCH_SELECTOR.keys(),
        help="Batch selector model.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the accuracy curve.",
    )

    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()
        random.seed(args.seed)

        df = format_csv(
            args.Train_file[0], config=args.config, verbose=args.verbose, add_missing=True
        )
        X_train, X_test, y_train, y_test = split_data(df, args.validation_ratio, args.seed)

        if args.accuracy:
            model = pd.read_csv(args.model, index_col=[0, 1])
        else:
            # model = train_model(
            #     X_train,
            #     y_train,
            #     args.learning_rate,
            #     args.epoch,
            #     batch_selector=BATCH_SELECTOR[args.batch],
            # )

            model = train(
                X_train,
                y_train,
                args.learning_rate,
                args.epoch,
                X_test=X_test if args.plot else None,
                Y_test=y_test if args.plot else None,
                batch_selector=BATCH_SELECTOR[args.batch],
            )

        y_pred = predict(X_test, model)
        y_pred = predict_house(y_pred)
        print(
            " the accuracy score is : ",
            "{:.2f}".format(accuracy_score(y_test.astype(int).to_list(), y_pred) * 100),
            "%",
        )

        if not args.accuracy:
            save_model(model, os.path.join(args.save_dir, args.model))
    except Exception as e:
        print("[ERROR] The training process failed")
        print("[ERROR] error: {}".format(e))
        raise ValueError("Error") from e
