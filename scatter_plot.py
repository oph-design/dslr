from libft import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math


colors = ["red", "orange", "blue", "green"]


def draw_scatterplot(data: pd.DataFrame, feature1: str, feature2: str, ax):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for i in range(len(houses)):
        y = data.loc[data["Hogwarts House"] == houses[i], feature2]
        x = data.loc[data["Hogwarts House"] == houses[i], feature1]
        ax.scatter(x, y, marker=".", color=colors[i], label=houses[i])
    ax.set_title(f"{feature1}/{feature2}")


def main():
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    features = list(data.columns[6:])
    size = len(features)
    fig = plt.figure(figsize=(4 * size, 2 * size))
    n = 1
    for i in range(size):
        for j in range(i):
            ax = fig.add_subplot(size, size, n)
            draw_scatterplot(data, features[i], features[j], ax)
            n += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
