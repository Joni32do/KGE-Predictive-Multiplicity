# Voting methods


import numpy as np

class VotingMethod:
    def __init__(self):
        pass

    def __call__(self, predicted_values_clf: np.ndarray) -> np.ndarray:
        """ Voting metod from social choice theory

        Expects the predicted values from multiple classifiers of
        shape (n_clf, n_queries, n_entities)
        for making a table n_queries=1 is required which automatically drops the dimension

        Returns:
            np.ndarray: _description_
        """
        raise NotImplementedError
    

class Majority(VotingMethod):
    def __call__(self, predicted_values: np.ndarray) -> np.ndarray:
        return np.argmax(predicted_values, axis=-1)

class Borda(VotingMethod):
    def __call__(self, predicted_values: np.ndarray) -> np.ndarray:
        # TODO:
        return (np.argsort(-predicted_values, axis=-1), np.arange(predicted_values.shape[-1]))

class Range(VotingMethod):
    def __call__(self, predicted_values: np.ndarray) -> np.ndarray:
        pass