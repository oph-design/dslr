from data_loader import check_input, format_data, houses
from gradient_descent import GradientDescent as GD
from stochastic_gd import Stochastic as SGD
from mini_batch import MiniBatch as MB
import pandas as pd
import sys

GREEN = "\033[92m"
RED = "\033[91m"
DEF = "\033[0m"

algorithms = ["classic", "stochastic", "mini-batch"]


def choose_algo(id: str, data: pd.DataFrame, house: str) -> GD:
    """selects alogorithm based on input"""
    if id not in algorithms:
        print(f"{RED}Wrong algorithm you need to enter: {algorithms}{DEF}")
        exit(1)
    if id == algorithms[1]:
        return SGD(data, house)
    if id == algorithms[2]:
        return MB(data, house)
    return GD(data, house)


def label_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """swapes house values out for 1s and 0s"""
    res = data.copy()
    res["Hogwarts House"] = (res["Hogwarts House"] == label).astype(int)
    return res


def main() -> None:
    """main function"""
    algorithm = "classic"
    if len(sys.argv) > 2:
        algorithm = sys.argv[2]
    data = check_input(sys.argv, 1)
    house_data = data["Hogwarts House"]
    data = format_data(data)
    data = pd.concat([house_data, data], axis=1)
    coefs = pd.DataFrame(columns=houses)
    for house in houses:
        training_data = label_data(data, house)
        model = choose_algo(algorithm, training_data, house)
        coefs[house] = model._train()
    coefs.to_csv("coefs.csv", index=False)
    print(f'{GREEN}Training Complete! Data written in "coefs.csv"{DEF}')


if __name__ == "__main__":
    main()
