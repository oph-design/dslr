import pandas as pd
import numpy as np
from tqdm import tqdm


class GradientDescent:

    def __init__(self, data: pd.DataFrame, house: str):
        self.house = house
        self.c = 0.0
        self.y = np.array(data.iloc[:, 0], dtype=np.float128)
        self.x = np.array(data.iloc[:, 1:], dtype=np.float128)
        self.m = np.zeros(self.x.shape[1])
        self.n = len(self.y)
        self.l = 0.001
        self.epochs = 12000

    def _predict(self) -> np.ndarray:
        scores = self.c + np.sum(self.m * self.x, axis=1)
        return 1 / (1 + np.exp(scores * -1))

    def _train(self):
        for i in tqdm(range(self.epochs), desc=self.house, ncols=80, ascii=True):
            predict = self._predict()
            cost_c = (1 / self.n) * np.sum(predict - self.y)
            self.c = self.c - self.l * cost_c
            for j in range(self.x.shape[1]):
                cost_m = (1 / self.n) * np.sum((predict - self.y) * self.x.T[j])
                self.m[j] = self.m[j] - self.l * cost_m
        return np.insert(self.m, 0, self.c)

