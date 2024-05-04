from data_loader import check_input, load_coefs, format_data, houses
from logreg_train import label_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

GREEN = "\033[92m"
DEF = "\033[0m"

colors = ["red", "green", "blue", "orange"]


def calculate_probs(data: np.ndarray, coefs: pd.DataFrame) -> np.ndarray:
    """calculates the propabilitys for all testcases"""
    res = []
    for house in houses:
        slicing = np.logical_not(np.isnan(coefs[house]))
        house_coefs = np.array(coefs[house][slicing])
        scores = house_coefs[0] + np.sum(house_coefs[1:] * data, axis=1)
        probs = 1 / (1 + np.exp(scores * -1))
        res.append(probs)
    return np.stack(res)


def write_result(probs: np.ndarray) -> None:
    """writes highest propabilities into file"""
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    for index, prob in enumerate(probs):
        house = houses[np.argmax(prob)]
        file.write(f"{index},{house}\n")
    file.close()
    print(f'{GREEN}Prediction Complete! Data written in "houses.csv"{DEF}')


def draw_graphs(data: pd.DataFrame, probs: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 6))
    for index, house in enumerate(houses):
        axes = fig.add_subplot(2, 2, index + 1)
        y = np.sort(label_data(data, house).iloc[:, 0])
        x = np.arange(0, len(y))
        axes.scatter(x, y, marker=".", color=colors[index])
        axes.plot(x, np.sort(probs[index]), color="k", linestyle=":")
        axes.set_title(house)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """main function"""
    data = check_input(sys.argv, 0)
    house_data = data["Hogwarts House"]
    data = format_data(data)
    coefs = load_coefs()
    probs = calculate_probs(np.array(data), coefs)
    write_result(probs.T)
    draw_graphs(pd.concat([house_data, data], axis=1), probs)


if __name__ == "__main__":
    main()
