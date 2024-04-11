import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import libft as ft


colors = ["red", "orange", "blue", "green"]


def calc_bins(arr):
    arr = arr[np.logical_not(np.isnan(arr))]
    width = 3.49 * ft.std(arr) / ft.count(arr) ** (1.0 / 3)
    return np.arange(ft.min(arr), ft.max(arr) + width, width)


def draw_histogram(data: pd.DataFrame, feature: str, ax):
    nbins = calc_bins(data[feature].to_numpy())
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for i in range(len(houses)):
        arr = data.loc[data["Hogwarts House"] == houses[i], feature]
        ax.hist(
            arr,
            bins=nbins,
            color=colors[i],
            label=houses[i],
            alpha=0.4,
            edgecolor=colors[i],
            linewidth=1.0,
        )
    ax.legend(loc="upper right")


def draw_all(data: pd.DataFrame):
    features = list(data.columns[6:])
    size = len(features)
    columns = math.ceil(math.sqrt(size))
    rows = math.ceil(size / columns)
    fig = plt.figure(figsize=(6 * columns, 3 * rows))
    for idx in range(size):
        ax = fig.add_subplot(rows, columns, idx + 1)
        draw_histogram(data, features[idx], ax)
        ax.set_title(features[idx])


def main():
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
