import pandas as pd
from models import GradientDescent as GD
from data_loader import check_input
import sys

GREEN = "\033[92m"
DEF = "\033[0m"

houses = {
    "Gryffindor": [
        "Hogwarts House",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
    "Slytherin": [
        "Hogwarts House",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
    "Ravenclaw": [
        "Hogwarts House",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
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
    # data = data.fillna(0)
    # numerical_columns = data.select_dtypes(include=["number"])
    # data[numerical_columns.columns] = numerical_columns.fillna(
    #     numerical_columns.median()
    numeric_columns = data.select_dtypes(include=['number'])
    means = numeric_columns.mean()
    stds = numeric_columns.std()
    normalized_numeric_df = (numeric_columns - means) / stds
    data = pd.concat([normalized_numeric_df, data.select_dtypes(exclude=['number'])], axis=1)
    print(data)
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
        model = GD(features, house)
        model._train()
        coefs[house] = model._getCoefs()
    coefs.to_csv("coefs.csv", index=False)
    print(f'{GREEN}Training Complete! Data written in "coefs.csv"{DEF}')


if __name__ == "__main__":
    main()
