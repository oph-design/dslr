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


def perform_regression(data: pd.DataFrame) -> np.ndarray:
    return np.array([])


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
    model = GD(label_data(data, "Gryffindor"))
    # for house in houses:
    #     coefs[house] = perform_regression(label_data(data, house))
    # coefs.to_csv("coefs.csv")


if __name__ == "__main__":
    main()
