import numpy as np
import pandas as pd
from data_loader import check_input, load_coefs
import sys

GREEN = "\033[92m"
DEF = "\033[0m"

houses = {
    "Gryffindor": [
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
    "Slytherin": [
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
    "Ravenclaw": [
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
    "Hufflepuff": [
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
}


def calculate_probs(data, coefs):
    res = []
    for house, subjects in houses.items():
        x_values = np.array(data.loc[:, subjects])
        house_coefs = np.array(coefs[house][np.logical_not(np.isnan(coefs[house]))])
        scores = house_coefs[0] + np.sum(house_coefs[1:] * x_values, axis=1)
        probs = 1 / (1 + np.exp(scores * -1))
        res.append(probs)
    return np.stack(res)


def write_result(probs):
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    for index, prob in enumerate(probs):
        house = list(houses.keys())[np.argmax(prob)]
        file.write(f"{index},{house}\n")
    file.close()
    print(f'{GREEN}Prediction Complete! Data written in "houses.csv"{DEF}')


def main():
    data = check_input(sys.argv, 0)
    # data = data.fillna(0)
    numeric_columns = data.select_dtypes(include=["number"])
    means = numeric_columns.mean()
    stds = numeric_columns.std()
    normalized_numeric_df = (numeric_columns - means) / stds
    data = pd.concat(
        [normalized_numeric_df, data.select_dtypes(exclude=["number"])], axis=1
    )
    coefs = load_coefs()
    probs = calculate_probs(data, coefs)
    write_result(probs.T)


if __name__ == "__main__":
    main()
