import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7

        # Adding a "border" to y_pred, nums cant be smaller than epsilon, or larger than 1 - epsilon 
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        L = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return round(L, 4)

    # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
    # y_pred: predicted probabilities (shape: n_samples x n_classes)
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        L = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(L, 4)
