import pandas as pd
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
    for house, subjects in houses.items():
        features = label_data(data.loc[:, subjects], house)
        model = GD(features)
        model._train()
        coefs[house] = model._getCoefs()
    coefs.to_csv("coefs.csv", index=False)


if __name__ == "__main__":
    main()
