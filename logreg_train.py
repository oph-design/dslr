import pandas as pd
import numpy as np
from models import GradientDescent as GD
from libft import check_input
import sys

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
features = ["Flying", "Divination", "Muggle Studies", "Charms"]


def label_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    res = data.copy()
    res["Hogwarts House"] = (res["Hogwarts House"] == label).astype(int)
    return res


def get_features(data: pd.DataFrame):
    return data.loc[
        :,
        [
            "Hogwarts House",
            "Flying",
            "Divination",
            "Muggle Studies",
            "Charms",
        ],
    ]


def main():
    data = get_features(check_input(sys.argv, 0))
    data = data.dropna()
    coefs = pd.DataFrame(
        columns=[
            "Gryffindor",
            "Slytherin",
            "Ravenclaw",
            "Hufflepuff",
        ]
    )
    for house, feature in zip(houses, features):
        model = GD(label_data(data, house), feature)
        model._train()
        coefs[house] = model._getCoefs()
    coefs.to_csv("coefs.csv", index=False)


if __name__ == "__main__":
    main()
