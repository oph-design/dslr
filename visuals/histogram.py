import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import libft as ft


def calc_bins(arr) -> np.ndarray:
    """calculates the number of bars"""
    arr = arr[np.logical_not(np.isnan(arr))]
    width = 3.49 * ft.std(arr) / ft.count(arr) ** (1.0 / 3)
    return np.arange(ft.min(arr), ft.max(arr) + width, width)


def draw_histogram(data: pd.DataFrame, feature: str, axes) -> None:
    """draws a histogram for one feature"""
    nbins = calc_bins(data[feature].to_numpy())
    for i in range(len(ft.houses)):
        arr = data.loc[data["Hogwarts House"] == ft.houses[i], feature]
        axes.hist(
            arr,
            bins=nbins,
            color=ft.colors[i],
            label=ft.houses[i],
            alpha=0.4,
            edgecolor=ft.colors[i],
            linewidth=1.0,
        )
    axes.legend(loc="upper right")


def draw_all(data: pd.DataFrame) -> None:
    """loops through all the features"""
    features = list(data.columns[6:])
    size = len(features)
    columns = math.ceil(math.sqrt(size))
    rows = math.ceil(size / columns)
    fig = plt.figure(figsize=(6 * columns, 3 * rows))
    for idx in range(size):
        axes = fig.add_subplot(rows, columns, idx + 1)
        draw_histogram(data, features[idx], axes)
        axes.set_title(features[idx])


def main() -> None:
    """main function"""
    data = ft.check_input(sys.argv, 1)
    if len(sys.argv) == 3:
        draw_histogram(data, sys.argv[2], plt)
        plt.title(sys.argv[2])
    else:
        draw_all(data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
