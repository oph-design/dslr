from data_loader import check_input, format_data, houses
import pandas as pd
from models import GradientDescent as GD
import sys

GREEN = "\033[92m"
DEF = "\033[0m"


def label_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    res = data.copy()
    res["Hogwarts House"] = (res["Hogwarts House"] == label).astype(int)
    return res


def main():
    data = check_input(sys.argv, 0)
    house_data = data["Hogwarts House"]
    data = format_data(data)
    data = pd.concat([house_data, data], axis=1)
    coefs = pd.DataFrame(
        columns=[
            "Gryffindor",
            "Slytherin",
            "Ravenclaw",
            "Hufflepuff",
        ]
    )
    for house in houses:
        model = GD(label_data(data, house), house)
        coefs[house] = model._train()
    coefs.to_csv("coefs.csv", index=False)
    print(f'{GREEN}Training Complete! Data written in "coefs.csv"{DEF}')


if __name__ == "__main__":
    main()
