import numpy as np
from typing import Iterable

from kge import KGE, KGE_dummy
from query import Query

class KGE_model():
    def __init__(self, entities: Iterable[str]):
        self.kge = KGE_dummy()
        self.entities = entities
        self.n_entities = len(entities)
        self.value_fn = lambda x: x

    def fit(self, X, y):
        self.X_train = X # (head, relation, tail)
        self.kge.fit(X, y)

    def predict_w_truth_prob(self, X: Iterable[Query],
                             truth_probs: Iterable[float],
                             elements_of_interest: Iterable[str]) -> np.ndarray:
        """ Makes the prediction according to truth_probs
        """
        assert len(X) == len(truth_probs) == len(elements_of_interest)

        predicted_values = np.zeros((len(X), self.n_entities))

        for i, query in enumerate(X):
            for j, e in enumerate(self.entities):
                if query.fill_in_missing_value(e) in self.X_train:
                    predicted_values[i, j] = self.value_fn(1.0)
                elif e is elements_of_interest[i]:
                    predicted_values[i, j] = self.value_fn(truth_probs[i])
                else:
                    predicted_values[i, j] = self.value_fn(1 - truth_probs[i])
    
        return predicted_values
    

    def top_k(self, predicted_values: np.ndarray,
                 X: Iterable[Query],
                 elements_of_interest: Iterable[str],
                 k: int) -> float:
        """ Computes the Top-K score for the given data
        predicted_values: (n_queries, n_entities)
        """
        y = np.zeros(len(X), dtype=bool)
        for i, e in enumerate(elements_of_interest):
            # Calculate the rank
            rank = np.argsort(-predicted_values[i])
            y[i] = np.where(rank == e)[0] < k
            print(e, rank)
        return y
    
    def hits_at_k(self, predicted_values: np.ndarray,
                  X: Iterable[Query],
                  elements_of_interest: Iterable[str],
                  k: int) -> float:
        """ Mean of Top-K
        """
        return self.top_k(predicted_values, X, elements_of_interest, k).mean()


class KGE_model_1(KGE_model):
    def __init__(self, entities: Iterable[str]):
        super().__init__(entities)
        self.value_fn = lambda x: 2*x+np.random.normal(0, 0.1)
        
class KGE_model_2(KGE_model):
    def __init__(self, entities: Iterable[str]):
        super().__init__(entities)
        self.value_fn = lambda x: 2*x+np.random.normal(0, 0.2)
        
class KGE_model_3(KGE_model):
    def __init__(self, entities: Iterable[str]):
        super().__init__(entities)
        self.value_fn = lambda x: 2*x+np.random.normal(0.1, 0.1)
        
class KGE_model_4(KGE_model):
    def __init__(self, entities: Iterable[str]):
        super().__init__(entities)
        self.value_fn = lambda x: 2*x+np.random.normal(-0.1, 0.1)
        
        


        

