import pandas as pd
import numpy as np


class GradientDescent:

    def __init__(self, data: pd.DataFrame, feature: str):
        self.c = 0.0
        self.m = 0.0
        self.y = np.array(data.iloc[:, 0], dtype=np.float128)
        self.x = np.array(data[feature], dtype=np.float128)
        self.n = len(self.y)
        self.l = 0.01
        self.epochs = 1000

    def _predict(self):
        predictions = self.m * self.x
        predictions = self.c + predictions
        return 1 / (1 + np.exp(predictions * -1))

    def _train(self):
        for i in range(self.epochs):
            prediction = self._predict()
            cost_c = (1 / self.n) * np.sum(prediction - self.y)
            self.c = self.c - self.l * cost_c
            cost_m = (1 / self.n) * np.sum((prediction - self.y) * self.x)
            self.m = self.m - self.l * cost_m

    def _getCoefs(self):
        return np.array([self.c, self.m])
