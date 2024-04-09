from libft import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

colors = ["red", "green", "blue", "yellow"]


def draw_hist(data: pd.DataFrame, feature: str):
    schools = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    histdata = np.array([])
    for i in range(len(schools)):
        grades = data.loc[data["Hogwarts House"] == schools[i], feature]
        histdata = np.concatenate((histdata, grades))
    plt.hist(histdata, bins=10, density=True, stacked=True)
    plt.show()


def main():
    data = load_data(sys.argv[1])
    if data is None:
        exit(1)
    print(data)
    draw_hist(data, "Arithmancy")


if __name__ == "__main__":
    main()
