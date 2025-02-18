"""Simple function ot manage configuration."""

import configparser
import json
import os


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
