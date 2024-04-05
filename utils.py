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


def ft_count(feature: np.ndarray):
    """returns array length"""
    return len(feature)


def ft_mean(feature: np.ndarray):
    """returns array mean"""
    return np.sum(feature) / ft_count(feature)


def ft_std(feature: np.ndarray):
    """returns array std deviation"""
    dev = feature - ft_mean(feature)
    sqsum = np.sum(np.power(dev, 2))
    return math.sqrt(sqsum / ft_count(feature))


def ft_min(feature: np.ndarray):
    """returns min value of array"""
    np.sort(feature)
    return feature[0]


def ft_max(feature: np.ndarray):
    """returns max value of array"""
    np.sort(feature)
    return feature[ft_count(feature) - 1]


def ft_percentile(feature: np.ndarray, percentile: float):
    """returns value at percentile"""
    if percentile < 0.0 and percentile > 1.0:
        raise Exception("percentile not between 0 and 1")
    pos = int(ft_count(feature) * percentile)
    return feature[pos]
