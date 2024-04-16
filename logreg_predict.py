import numpy as np
import libft as ft
import pandas as pd
import sys

RED = "\033[91m"
DEF = "\033[0m"
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def load_coefs() -> pd.DataFrame:
    try:
        coefs = pd.read_csv("coefs.csv")
        headers = list(coefs)
        if len(headers) != 4 or len(coefs.iloc[:, 1]) != 5:
            raise Exception("Wrong Format")
        return coefs
    except Exception:
        coefs = {
            "Gryffindor": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Slytherin": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Ravenclaw": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Hufflepuff": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return pd.DataFrame(coefs)


def calculate_probs(data, coefs):
    scores = np.dot(data, coefs.T)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def write_result(probs):
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    for index, prob in enumerate(probs):
        house = houses[prob.index(ft.max(prob))]
        file.write(f"{index},{house}\n")
    file.close()


def main():
    data = ft.check_input(sys.argv, 0)
    coefs = load_coefs()
    data = data.loc[
        :,
        [
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Ancient Runes",
            "Charms",
        ],
    ]
    probs = calculate_probs(np.array(data), np.array(coefs))
    write_result(probs)


if __name__ == "__main__":
    main()
