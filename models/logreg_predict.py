import numpy as np
import libft as ft
import pandas as pd
import sys

RED = "\033[91m"
DEF = "\033[0m"
houses = {
    "Gryffindor": ["Flying", "Transfiguration", "History of Magic"],
    "Slytherin": ["Divination"],
    "Ravenclaw": ["Muggle Studies", "Charms"],
    "Hufflepuff": [
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
    ],
}


def load_coefs() -> pd.DataFrame:
    try:
        coefs = pd.read_csv("coefs.csv")
        headers = list(coefs)
        if len(headers) != 4:
            raise Exception("Wrong Format")
        return coefs
    except Exception:
        coefs = {
            "Gryffindor": [0.0, 0.0, 0.0, 0.0],
            "Slytherin": [0.0, 0.0, np.nan, np.nan],
            "Ravenclaw": [0.0, 0.0, 0.0, np.nan],
            "Hufflepuff": [0.0, 0.0, 0.0, 0.0],
        }
        return pd.DataFrame(coefs)


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
    print(probs)
    for index, prob in enumerate(probs):
        house = list(houses.keys())[np.argmax(prob)]
        file.write(f"{index},{house}\n")
    file.close()


def main():
    data = ft.check_input(sys.argv, 0)
    data = data.fillna(0)
    coefs = load_coefs()
    probs = calculate_probs(data, coefs)
    write_result(probs.T)


if __name__ == "__main__":
    main()
