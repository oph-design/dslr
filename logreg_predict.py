import numpy as np
import libft as ft
import pandas as pd
import sys

RED = "\033[91m"
DEF = "\033[0m"
houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
features = ["Flying", "Divination", "Muggle Studies", "Charms"]


def load_coefs() -> pd.DataFrame:
    try:
        coefs = pd.read_csv("coefs.csv")
        headers = list(coefs)
        if len(headers) != 4 or len(coefs.iloc[:, 1]) != 2:
            raise Exception("Wrong Format")
        return coefs
    except Exception:
        coefs = {
            "Gryffindor": [0.0, 0.0],
            "Slytherin": [0.0, 0.0],
            "Ravenclaw": [0.0, 0.0],
            "Hufflepuff": [0.0, 0.0],
        }
        return pd.DataFrame(coefs)


def calculate_probs(data, coefs):
    res = []
    for feature, coef in zip(features, coefs):
        scores = np.array(coef[0] + data[feature] * coef[1])
        probs = 1 / (1 + np.exp(scores * -1))
        res.append(probs)
    return np.stack(res)


def write_result(probs):
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    for index, prob in enumerate(probs):
        house = houses[np.argmax(prob)]
        file.write(f"{index},{house}\n")
    file.close()


def main():
    data = ft.check_input(sys.argv, 0)
    coefs = load_coefs()
    probs = calculate_probs(data, np.array(coefs).T)
    write_result(probs.T)


if __name__ == "__main__":
    main()
