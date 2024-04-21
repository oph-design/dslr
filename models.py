import pandas as pd
import numpy as np
import libft as ft


class GradientDescent:

    def __init__(self, data: pd.DataFrame):
        self.c = 0.0
        self.m = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float128)
        self.y = np.array(data.iloc[:, 1], dtype=np.float128)
        self.x = np.array(data.iloc[:, 1:6], dtype=np.float128)
        self.n = len(self.y)
        self.l = 0.01
        self.epochs = 1000
        self._normalize()

    def _reset(self, data: pd.DataFrame):
        self.c = 0.0
        self.m = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float128)
        self.y = np.array(data.iloc[:, 1], dtype=np.float128)

    def _normalize(self):
        for i in range(len(self.x)):
            mean = ft.mean(self.x[i])
            std = ft.std(self.x[i])
            self.x[i] = (self.x[i] - mean) / std

    def _predict(self):
        predictions = self.m * self.x
        predictions = self.c + np.sum(predictions, axis=1)
        return 1 / (1 + np.exp(predictions * -1))

    def _updateSlopes(self, prediction):
        for j in range(5):
            cost = (1 / self.n) * np.sum((prediction - self.y) * self.x[:, j])
            self.m[j] = self.m[j] - self.l * cost

    def _train(self):
        for i in range(self.epochs):
            prediction = self._predict()
            cost_c = (1 / self.n) * np.sum(prediction - self.y)
            self.c = self.c - self.l * cost_c
            self._updateSlopes(prediction)

    def _getCoefs(self):
        return np.insert(self.m, 0, self.c)
