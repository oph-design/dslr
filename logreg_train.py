import pandas as pd
import numpy as np
from libft import check_input
import sys

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

class model:
    

    def __init__(self, data: pd.Dataframe):
        self.c = 0
        self.m = [0, 0, 0, 0, 0]
        self.y = data.iloc[:, 1]
        self.x = data
        self.l = 0.01
        self.epochs = 1000

    def _predict(self):
        prediction = self.c \
                    + self.m[0] * self.data["Astronomy"].values \
                    + self.m[1] * self.data["Herbology"].values \
                    + self.m[2] * self.data["Defense Against the Dark Arts"] \
                    + self.m[3] * self.data["Ancient Runes"] \
                    + self.m[4] * self.data["Charms"] 
        return prediction

    def _train(self):
        for i in range(epochs):
            prediction = self._predict()  


def label_data(data: pd.DataFrame,label: str) -> pd.DataFrame:
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


def calc_loss(x, y, length, slope) -> float:
    prediction = C + M * x
    if slope is False:
        return (2 / length) * sum(prediction - y)
    return (2 / length) * sum(x * (prediction - y))


def train_model(x, y, iterations):
    global C, M
    length = len(x)
    plot_data(x, y, "Graph at Start", 1)
    for i in range(iterations):
        loss_c = calc_loss(x, y, length, False)
        loss_m = calc_loss(x, y, length, True)
        # print(f"!Update! theta0 = {C} theta1 = {M}")
        plot_data(x, y, f"Graph Progression Epoch:{i}", 0.01)
        C = C - L * loss_c
        M = M - L * loss_m


def perform_regression(data: pd.DataFrame) -> np.ndarray:
    return []

def main():
    data = get_features(check_input(sys.argv, 0))
    coefs = pd.DataFrame(columns=[
        "Gryffindor",
        "Slytherin",
        "Ravenclaw",
        "Hufflepuff",
    ])
    for house in houses:
        coefs[house] = perform_regression(label_data(data, house))
    coefs.to_csv("coefs.csv")
    
    


if __name__ == "__main__":
    main()
