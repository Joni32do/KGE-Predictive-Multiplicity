# Predict the missing links of a network using a KGE model proxy (due to time reasons artificial data is used)
#
# Voting methods from social choice theory, i.e. Ensemble learning
#
# Prints a latex table with the results and

import numpy as np

from typing import Iterable


from kge import KGE_dummy
from kge_models import *
from query import Query
from voting_methods import Majority, Borda, Range
from plot_graphs import plot_graph


def latex_table(self, X: Iterable[Query],
                elements_of_interest: Iterable[str],
                predicted_values: np.ndarray,
                kge_models,
                voting_methods) -> str:
    """ Converts the predicted values to a latex table

    Input:
        model_preds.shape = (n_models, n_entities)
    """
    table = r"\\begin{tabular}{}"
    table += f"{|c| for _ in range(len(elements_of_interest))}"


def main(entities, train_relations):
    # Prediction what orbits the sun
    test_queries = [Query("Sun", "orbits", head_is_missing=True)]

    # Define Models and Voting methods
    kge_models = [KGE_model_1(entities),
                  KGE_model_2(entities),
                  KGE_model_3(entities),
                  KGE_model_4(entities)]
        
    voting_methods = [Majority(), Borda(), Range()]
    
    # "Training" (just memorizing)
    for model in kge_models:
        model.fit(train_relations, [0.] * len(train_relations))

    # 
    model_preds = np.zeros((len(kge_models), len(test_queries), len(entities)))
    # Prediction
    for i, model in enumerate(kge_models):
        model_pred = model.predict_w_truth_prob(test_queries, [0.4], ["Moon"])
        idz = np.argsort(-model.predict_w_truth_prob(test_queries, [0.4], ["Moon"]))
    
    # Since only one query for table, remove inner dim
    # (n_models, n_queries, n_entities) -> (n_models, n_entities)
    model_preds = model_preds.squeeze()
    latex_str = latex_table()
    

if __name__ == "__main__":
    # Draw nodes (entities)
    entities_dict = {
        "Sun": (0, 0),
        "Mars": (3, 2),
        "Jupiter": (-3, 1),
        "Titan": (-1.5, 2),
        "James Webb": (0, -2),
        "Earth": (2.5, -1.5),
        "Curiosity Rover": (4, 0),
        "Moon": (0, 1.5),
        "Hubble": (-1.5, -1.5),
        "Sirius": (-3, -0.5)
    }
    entities = list(entities_dict.keys())
    print(entities)

    # Triplets (start, end, relation, truth_prob)
    train_relations = [
        ("Earth", "orbits", "Sun"),
        ("Mars", "orbits", "Sun"),
        ("Moon", "orbits", "Earth"),
        ("Moon", "orbits", "Sun"),
        ("Jupiter", "orbits", "Sun"),
        ("James Webb", "orbits", "Earth"),
        ("James Webb", "observes", "Jupiter"),
    ]
    test_relations = [
        ("Earth", "observes", "Curiosity Rover", 0.9),
        ("Curiosity Rover", "observes", "Mars", 0.9),
        ("Curiosity Rover", "orbits", "Sun", 0.9),
        ("Sun", "orbits", "James Webb", 0.1),
    ]

    plot_graph(entities_dict, train_relations, test_relations)
    main(entities, train_relations)






        