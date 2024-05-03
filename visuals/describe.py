import sys
import numpy as np
import pandas as pd
import libft as ft


rows = [
    "Count",
    "Mean",
    "Median",
    "Mode",
    "Std",
    "Var",
    "Min",
    "25%",
    "50%",
    "75%",
    "Max",
]


def transform(feature: np.ndarray) -> np.ndarray | None:
    """returns array with data values"""
    if feature.dtype != np.int64 and feature.dtype != np.float64:
        return None
    feature = feature[np.logical_not(np.isnan(feature))]
    if ft.count(feature) == 0:
        return None
    return np.array(
        [
            ft.count(feature),
            ft.mean(feature),
            ft.percentile(feature, 0.5),
            ft.mode(feature),
            ft.std(feature),
            ft.variance(feature),
            ft.min(feature),
            ft.percentile(feature, 0.25),
            ft.percentile(feature, 0.5),
            ft.percentile(feature, 0.75),
            ft.max(feature),
        ]
    )


def ft_describe(data: pd.DataFrame) -> None:
    """prints description grid"""
    df = pd.DataFrame({"": rows})
    columns = list(data.columns)
    for i in range(data.shape[1]):
        feature = transform(np.array(data.iloc[:, i]))
        if feature is not None:
            df[columns[i]] = feature
    print(df.to_string(index=False))


def main() -> None:
    """main function"""
    data = ft.check_input(sys.argv, 0)
    ft_describe(data)


if __name__ == "__main__":
    main()
