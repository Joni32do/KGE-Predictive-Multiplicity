from typing import Iterable
import numpy as np

class KGE():
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(words: Iterable[str]) -> np.ndarray:
        pass
    
class KGE_dummy(KGE):
    def predict(words: Iterable[str]) -> np.ndarray:
        return np.arange(len(words))