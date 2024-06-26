import pandas as pd
import numpy as np
import math
import sys


RED = "\033[91m"
DEF = "\033[0m"

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
colors = ["red", "orange", "blue", "green"]


def count(feature: np.ndarray):
    """returns array length"""
    return len(feature)


def mean(feature: np.ndarray):
    """returns array mean"""
    return np.sum(feature) / count(feature)


def mode(feature: np.ndarray):
    """returns array mode"""
    values = []
    counts = []
    for value in feature:
        if values not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[values.index(values)] += 1
    max_count = max(np.array(counts))
    return values[counts.index(max_count)]


def variance(feature: np.ndarray):
    """returns arrays variance"""
    dev = feature - mean(feature)
    sqsum = np.sum(np.power(dev, 2))
    return sqsum / (count(feature) - 1)


def std(feature: np.ndarray):
    """returns array std deviation"""
    return math.sqrt(variance(feature))


def min(feature: np.ndarray):
    """returns min value of array"""
    sorted = np.sort(feature)
    return sorted[0]


def max(feature: np.ndarray):
    """returns max value of array"""
    sorted = np.sort(feature)
    return sorted[count(feature) - 1]


def percentile(feature: np.ndarray, percentile: float):
    """returns value at percentile"""
    if percentile < 0.0 and percentile > 1.0:
        raise Exception("percentile not between 0 and 1")
    sorted = np.sort(feature)
    pos = math.floor(count(sorted) * percentile)
    if percentile > 0.5:
        pos += 1
    return sorted[pos]


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
