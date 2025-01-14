import argparse
import json
import os

from logreg_predict import predict, predict_house
from sklearn.metrics import accuracy_score

from ft_dslr.logistic_regression import train_model
from ft_dslr.logistic_regression.tools import format_csv, split_data


def save_model(model: dict, file_name: str) -> None:
    """
    Save the model as a json file.
    :param model: Weights to save.
    :param file_name: Name of the file.
    :return: None
    """
    with open(file_name, "w") as file:
        json.dump(model, file, indent=4)


def options_parser():
    """Use to handle program parameters and options."""
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
    parser.add_argument("-m", "--model", type=str, default="model.json", help="The model file.")
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

    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()
        df = format_csv(args.Train_file[0], config=args.config, verbose=args.verbose)
        X_train, X_test, y_train, y_test = split_data(df, args.validation_ratio, args.seed)

        if args.accuracy:
            model = json.load(open(args.model))
        else:
            model = train_model(X_train, y_train, args.learning_rate, args.epoch)

        y_pred = predict(X_test, model)
        y_pred = predict_house(y_pred)
        print(
            " the accuracy score is : ",
            "{:.2f}".format(accuracy_score(y_test.astype(int).to_list(), y_pred) * 100),
            "%",
        )

        if not args.accuracy:
            save_model(model, os.path.join(args.save_dir, "model.json"))
    except Exception as e:
        print("[ERROR] The training process failed")
        print("[ERROR] error: {}".format(e))
