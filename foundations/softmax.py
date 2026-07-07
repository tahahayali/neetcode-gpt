import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        stabilized_exp = z - max(z)
        numerator = np.exp(stabilized_exp)
        denominator = np.sum(np.exp(stabilized_exp))
        score = numerator / denominator
        return np.round(score, 4)