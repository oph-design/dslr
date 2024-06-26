import pandas as pd
import numpy as np
import sys


RED = "\033[91m"
DEF = "\033[0m"

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
features = [
    "Ancient Runes",
    "Defense Against the Dark Arts",
    "Herbology",
]


def load_coefs() -> pd.DataFrame:
    try:
        coefs = pd.read_csv("coefs.csv")
        headers = list(coefs)
        if len(headers) != 4:
            raise Exception("Wrong Format")
        return coefs
    except Exception:
        coefs = {
            "Gryffindor": [0.0, 0.0, 0.0, 0.0],
            "Slytherin": [0.0, 0.0, 0.0, 0.0],
            "Ravenclaw": [0.0, 0.0, 0.0, 0.0],
            "Hufflepuff": [0.0, 0.0, 0.0, 0.0],
        }
        return pd.DataFrame(coefs)


def checker(argv: list, argc: int) -> pd.DataFrame:
    """checks the input for validity and returns it"""
    if len(argv) < 2 or len(argv) > argc + 2:
        raise Exception("Wrong number of Arguments provided")
    data = pd.read_csv(argv[1])
    columns = list(data.columns)
    if len(columns) < 6 + argc:
        raise Exception("CSV has not enough features")
    return data


def check_input(argv: list, argc: int) -> pd.DataFrame:
    """catches occuring Exceptions from checker"""
    try:
        return checker(argv, argc)
    except Exception as e:
        print(
            f"{RED}Program terminated because of Exception:\n{str(e)}{DEF}",
            file=sys.stderr,
        )
        exit(1)


def format_data(data: pd.DataFrame):
    data = data.loc[:, features]
    numeric_columns = data.select_dtypes(include=["number"])
    means = numeric_columns.mean()
    stds = numeric_columns.std()
    normalized_numeric_df = (numeric_columns - means) / stds
    data = pd.DataFrame(normalized_numeric_df)
    return data.fillna(0)
