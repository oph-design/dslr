from libft import load_data
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({"font.size": 6})
colors = ["red", "orange", "blue", "green"]


def draw_scatterplot(data: pd.DataFrame, feat: str, comp: str, axes):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for i in range(len(houses)):
        y = data.loc[data["Hogwarts House"] == houses[i], feat]
        x = data.loc[data["Hogwarts House"] == houses[i], comp]
        axes.scatter(x, y, marker=".", color=colors[i], label=houses[i])
    axes.set_xlabel(comp)
    axes.set_ylabel(feat)


def main():
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    n = 1
    features = list(data.columns[6:])
    compare = features[1:]
    size = len(features)
    fig = plt.figure(figsize=(3 * size, 1 * size))
    for feat in features:
        for i, comp in enumerate(compare):
            rowlen = len(compare)
            axes = fig.add_subplot(size, size, n)
            draw_scatterplot(data, feat, comp, axes)
            n = n + 1 if i != rowlen - 1 else n + size - i
        compare = compare[1:]
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, wspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()
