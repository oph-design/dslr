import sys
import numpy as np
import pandas as pd
from utils import ft_count, ft_max, ft_mean, ft_min, ft_percentile, ft_std, load_data


def transform(feature: np.ndarray) -> np.ndarray | None:
    if feature.dtype != np.int64 and feature.dtype != np.float64:
        return None
    return np.array(
        [
            ft_count(feature),
            ft_mean(feature),
            ft_std(feature),
            ft_min(feature),
            ft_percentile(feature, 0.25),
            ft_percentile(feature, 0.5),
            ft_percentile(feature, 0.75),
            ft_max(feature),
        ]
    )


def ft_describe(data: pd.DataFrame):
    rows = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    df = pd.DataFrame({"": rows})
    columns = list(data.columns)
    for i in range(data.shape[1]):
        feature = transform(data.iloc[:, i])
        if feature is not None:
            df[columns[i]] = feature
    print(df)


def main():
    if len(sys.argv) < 2:
        exit(1)
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    # print(data.describe())
    print(data)
    # ft_describe(data)


if __name__ == "__main__":
    main()
