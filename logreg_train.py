import pandas as pd
import numpy as np
from models import GradientDescent as GD
from libft import check_input
import sys

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]


def label_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    data["Hogwarts House"] = (data["Hogwarts House"] == label).astype(int)
    return data


def get_features(data: pd.DataFrame):
    return data.loc[
        :,
        [
            "Hogwarts House",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Ancient Runes",
            "Charms",
        ],
    ]


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    nan_indices = data.iloc[:, -5:].isnull().any(axis=1)
    return pd.DataFrame(data[~nan_indices])


def main():
    data = get_features(check_input(sys.argv, 0))
    coefs = pd.DataFrame(
        columns=[
            "Gryffindor",
            "Slytherin",
            "Ravenclaw",
            "Hufflepuff",
        ]
    )
    for house in houses:
        model = GD(remove_nans(label_data(data, house)))
        model._train()
        coefs[house] = model._getCoefs()
    coefs.to_csv("coefs.csv")


if __name__ == "__main__":
    main()
