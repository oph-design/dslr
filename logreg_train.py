import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import GradientDescent as GD
from libft import check_input
import sys

houses = {
    "Gryffindor": ["Hogwarts House", "Flying", "Transfiguration", "History of Magic"],
    "Slytherin": ["Hogwarts House", "Divination"],
    "Ravenclaw": ["Hogwarts House", "Muggle Studies", "Charms"],
    "Hufflepuff": [
        "Hogwarts House",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
}


def label_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    res = data.copy()
    res["Hogwarts House"] = (res["Hogwarts House"] == label).astype(int)
    return res


def main():
    data = check_input(sys.argv, 0)
    data = data.dropna()
    coefs = pd.DataFrame(
        columns=[
            "Gryffindor",
            "Slytherin",
            "Ravenclaw",
            "Hufflepuff",
        ]
    )
    i = 0
    fig = plt.figure(figsize=(6 * 2, 3 * 2))
    for house, subjects in houses.items():
        i += 1
        features = label_data(data.loc[:, subjects], house)
        axes = fig.add_subplot(2, 2, i)
        axes.scatter(
            np.arange(0, len(features.iloc[:, 0])),
            np.sort(features.iloc[:, 0]),
            marker=".",
        )
        model = GD(features)
        model._train()
        coefs_h = model._getCoefs()
        scores = coefs_h[0] + np.sum(coefs_h[1:] * features.iloc[:, 1:], axis=1)
        probs = 1 / (1 + np.exp(scores * -1))
        axes.plot(np.arange(0, len(features.iloc[:, 0])), np.sort(probs), color="g")
        axes.set_title(house)
        # coefs[house] = model._getCoefs()
    # coefs.to_csv("coefs.csv", index=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
