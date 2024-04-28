import libft as ft
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math

data = pd.DataFrame([])
columns = 0
rows = 0
n = 1


def draw_scatterplot(feat: str, comp: str, axes) -> None:
    """draws scatter plot between 2 feature"""
    for i in range(len(ft.houses)):
        y = data.loc[data["Hogwarts House"] == ft.houses[i], feat]
        x = data.loc[data["Hogwarts House"] == ft.houses[i], comp]
        axes.scatter(x, y, marker=".", color=ft.colors[i], label=ft.houses[i])
    if n == 1:
        plt.legend(loc="upper left")


def draw_row(compare: list, feature: str, fig) -> None:
    """loops through all features once"""
    global n
    for i, comp in enumerate(compare):
        rowlen = len(compare)
        axes = fig.add_subplot(rows, columns, n)
        draw_scatterplot(feature, comp, axes)
        axes.set_xlabel(comp)
        axes.set_ylabel(feature)
        n = n + 1 if i != rowlen - 1 else n + rows - i
    fig.tight_layout()


def draw_all(features: list) -> None:
    """loops through all features for every feature"""
    plt.rcParams.update({"font.size": 6})
    compare = features[1:]
    fig = plt.figure(figsize=(3 * columns, 1 * rows))
    for feat in features:
        draw_row(compare, feat, fig)
        compare = compare[1:]


def main() -> None:
    """main function"""
    global data, rows, columns
    data = ft.check_input(sys.argv, 2)
    features = list(data.columns[6:])
    rows = columns = len(features)
    if len(sys.argv) == 4:
        draw_scatterplot(sys.argv[2], sys.argv[3], plt)
        plt.xlabel(sys.argv[3])
        plt.ylabel(sys.argv[2])
    elif len(sys.argv) == 3:
        features.remove(sys.argv[2])
        rows = math.floor(math.sqrt(len(features)))
        columns = math.ceil(len(features) / rows)
        fig = plt.figure(figsize=(4 * columns, 2 * rows))
        draw_row(features, sys.argv[2], fig)
    else:
        draw_all(features)
    plt.subplots_adjust(left=0.1, wspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()
