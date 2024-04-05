import pandas as pd
import numpy as np
import math


def load_data(name: str) -> pd.DataFrame | None:
    """secures the read data fits the requirements"""
    try:
        return pd.read_csv(name)
    except FileNotFoundError:
        print("Read CSV: File not found.")
    except pd.errors.EmptyDataError:
        print("Read CSV: No data")
    except pd.errors.ParserError:
        print("Read CSV: Parse error")
    except BaseException:
        print("Read CSV: unexpected error")
    return None


def count(feature: np.ndarray):
    """returns array length"""
    return len(feature)


def mean(feature: np.ndarray):
    """returns array mean"""
    return np.sum(feature) / count(feature)


def std(feature: np.ndarray):
    """returns array std deviation"""
    dev = feature - mean(feature)
    sqsum = np.sum(np.power(dev, 2))
    return math.sqrt(sqsum / count(feature))


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
    pos = int(count(sorted) * percentile)
    return sorted[pos]
