import sys
import numpy as np
import pandas as pd
import libft as ft


def transform(feature: np.ndarray) -> np.ndarray | None:
    if feature.dtype != np.int64 and feature.dtype != np.float64:
        return None
    feature = feature[np.logical_not(np.isnan(feature))]
    return np.array(
        [
            ft.count(feature),
            ft.mean(feature),
            ft.std(feature),
            ft.min(feature),
            ft.percentile(feature, 0.25),
            ft.percentile(feature, 0.5),
            ft.percentile(feature, 0.75),
            ft.max(feature),
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
    print(df.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        exit(1)
    data = ft.load_data(sys.argv[1])
    if data is None:
        exit(1)
    print(data.describe())
    ft_describe(data)


if __name__ == "__main__":
    main()
