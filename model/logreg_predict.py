from data_loader import check_input, load_coefs, format_data, houses
import pandas as pd
import numpy as np
import sys

GREEN = "\033[92m"
DEF = "\033[0m"


def calculate_probs(data: np.ndarray, coefs: pd.DataFrame) -> np.ndarray:
    """calculates the propabilitys for all testcases"""
    res = []
    for house in houses:
        slicing = np.logical_not(np.isnan(coefs[house]))
        house_coefs = np.array(coefs[house][slicing])
        scores = house_coefs[0] + np.sum(house_coefs[1:] * data, axis=1)
        probs = 1 / (1 + np.exp(scores * -1))
        res.append(probs)
    return np.stack(res)


def write_result(probs: np.ndarray) -> None:
    """writes highest propabilities into file"""
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    for index, prob in enumerate(probs):
        house = houses[np.argmax(prob)]
        file.write(f"{index},{house}\n")
    file.close()
    print(f'{GREEN}Prediction Complete! Data written in "houses.csv"{DEF}')


def main() -> None:
    """main function"""
    data = check_input(sys.argv, 0)
    data = format_data(data)
    coefs = load_coefs()
    probs = calculate_probs(np.array(data), coefs)
    write_result(probs.T)


if __name__ == "__main__":
    main()
