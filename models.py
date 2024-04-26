import pandas as pd
import numpy as np


class GradientDescent:

    def __init__(self, data: pd.DataFrame):
        self.c = 0.0
        self.y = np.array(data.iloc[:, 0], dtype=np.float128)
        self.n = len(self.y)
        self.m = np.zeros(self.n)
        self.x = np.array(data.iloc[:, 1:], dtype=np.float128)
        self.l = 0.01
        self.epochs = 1000

    def _predict(self):
        predictions = self.m * self.x
        predictions = self.c + predictions
        return 1 / (1 + np.exp(predictions * -1))

    def _train(self):
        for i in range(self.epochs):
            predict = self._predict()
            cost_c = (1 / self.n) * np.sum(predict - self.y)
            self.c = self.c - self.l * cost_c
            for j in range(self.n):
                cost_m = (1 / self.n) * np.sum((predict - self.y) * self.x[j])
                self.m[j] = self.m[j] - self.l * cost_m

    def _getCoefs(self):
        return np.insert(self.m, 0, self.c)
