from gradient_descent import GradientDescent
import numpy as np
import pandas as pd


class MiniBatch(GradientDescent):

    def __init__(self, data: pd.DataFrame, house: str):
        super().__init__(data, house)
        self.defaultx = self.x
        self.defaulty = self.y

    def _update_sample(self):
        self.x = []
        self.y = []
        for i in range(self.n // 100):
            index = np.random.randint(0, self.n)
            self.x.append(self.defaultx[index])
            self.y.append(self.defaulty[index])
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def _predict(self) -> np.ndarray:
        self._update_sample()
        scores = self.c + np.sum(self.m * self.x, axis=1)
        return 1 / (1 + np.exp(scores * -1))
