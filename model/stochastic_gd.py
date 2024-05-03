from gradient_descent import GradientDescent
import numpy as np
import pandas as pd


class Stochastic(GradientDescent):

    def __init__(self, data: pd.DataFrame, house: str):
        super().__init__(data, house)
        self.defaultx = self.x
        self.defaulty = self.y

    def _update_sample(self):
        index = np.random.randint(0, self.n)
        self.x = self.defaultx[index]
        self.y = self.defaulty[index]

    def _predict(self) -> np.ndarray:
        self._update_sample()
        scores = self.c + np.sum(self.m * self.x)
        return 1 / (1 + np.exp(scores * -1))
