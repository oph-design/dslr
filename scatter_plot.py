from libft import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math


colors = ["red", "orange", "blue", "green"]


def draw_scatterplot(data: pd.DataFrame, feature: str, ax):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for i in range(len(houses)):
        y = data.loc[data["Hogwarts House"] == houses[i], feature]
        y = y[np.logical_not(np.isnan(y))]
        ax.scatter(y.index, y, marker=".", color=colors[i], label=houses[i])
    ax.legend(loc="upper right")
    ax.set_title(feature)


def main():
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    features = list(data.columns[6:])
    size = len(features)
    columns = math.ceil(math.sqrt(size))
    rows = math.ceil(size / columns)
    fig = plt.figure(figsize=(6 * columns, 3 * rows))
    for idx in range(size):
        ax = fig.add_subplot(rows, columns, idx + 1)
        draw_scatterplot(data, features[idx], ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
