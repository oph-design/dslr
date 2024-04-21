import pandas as pd
import numpy as np


class GradientDescent:

    def __init__(self, data: pd.DataFrame):
        self.c = 0
        self.m = np.array([0, 0, 0, 0, 0])
        self.y = data.iloc[:, 1]
        self.x = data.iloc[:, 1:6]
        self.n = len(self.y)
        self.l = 0.01
        self.epochs = 1000

    def _predict(self):
        predictions = self.m * self.x
        print(np.shape(predictions))
        predictions = self.c + np.sum(predictions, axis=1)
        return 1 / (1 + np.exp(predictions * -1))

    def _updateSlopes(self, prediction):
        for j in range(len(self.m)):
            cost = (1 / self.n) * np.sum((prediction - self.y) * self.x[j])
            self.m[j] = self.m[j] - self.l * cost

    def _train(self):
        for i in range(self.epochs):
            prediction = self._predict()
            cost_c = (1 / self.n) * np.sum(prediction - self.y)
            self.c = self.c - self.l * cost_c
            self._updateSlopes(prediction)

    def _getCoefs(self):
        return np.concatenate((self.c, self.m))
