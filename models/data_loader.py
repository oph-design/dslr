import pandas as pd
import numpy as np
import sys


RED = "\033[91m"
DEF = "\033[0m"


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
            "Slytherin": [0.0, 0.0, np.nan, np.nan],
            "Ravenclaw": [0.0, 0.0, 0.0, np.nan],
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
    for i in range(argc):
        if i + 2 >= len(argv):
            break
        try:
            columns.index(argv[i + 2])
        except ValueError:
            raise Exception("Feature(s) not found in data")
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
